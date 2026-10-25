/** UI logic for the fully client-side tool. All work happens in worker.js. */

const state = {
  info: null,
  file: null,
  sourceUrl: null,
  axes: new Set(["wght"]),
  deltaLevel: 1,
  preset: "latin",
  custom: "",
  estimate: null,
  generating: false,
  autoStart: false,
  result: null,
  fontFace: null,
  fontUrl: null,
  modelReady: false,
};

const $ = (id) => document.getElementById(id);
const nf = new Intl.NumberFormat("he-IL");

const worker = new Worker("worker.js", { type: "module" });
const pending = new Map();
let requestId = 0;

const uploadError = $("upload-error");
const jobError = $("job-error");
const generateBtn = $("generate-btn");
const generateNote = $("generate-note");
const estimateText = $("estimate-text");
const customRange = $("custom-range");
const progressWrap = $("progress-wrap");
const progressBar = $("progress-bar");
const progressMessage = $("progress-message");
const progressEta = $("progress-eta");
const resultBody = $("result-body");
const allAxesBtn = $("all-axes-btn");
const specimen = $("specimen");
const slidersEl = $("sliders");
const settingsCard = $("settings-card");
const resultCard = $("result-card");
const fontCard = $("font-card");
const dropzone = $("dropzone");
const fileInput = $("file-input");

function request(type, payload = {}, transfer = []) {
  const id = ++requestId;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject, type });
    worker.postMessage({ ...payload, type, id }, transfer);
  });
}

worker.onmessage = (event) => {
  const message = event.data || {};
  if (message.type === "model-progress") {
    const percent = Math.round((message.fraction || 0) * 100);
    setModelStatus(`מוריד את המודל… ${percent}%`, "loading");
    return;
  }
  if (message.type === "model-ready") {
    state.modelReady = true;
    setModelStatus("המודל מוכן", "ready");
    return;
  }
  if (message.type === "model-error") {
    setModelStatus("המודל לא נטען", "error");
    showError(uploadError, `טעינת המודל נכשלה: ${message.message}`);
    return;
  }
  if (message.type === "progress") {
    updateProgress(message);
    return;
  }
  const entry = pending.get(message.id);
  if (!entry) return;
  pending.delete(message.id);
  if (message.type === "error") entry.reject(new Error(message.message));
  else entry.resolve(message);
};

function setModelStatus(text, variant) {
  $("model-status-text").textContent = text;
  const dot = $("model-dot");
  dot.className = "dot";
  if (variant === "ready") dot.classList.add("ok");
  if (variant === "error") dot.classList.add("bad");
  $("model-status").classList.toggle("is-busy", variant === "loading");
}

function showError(element, message) {
  element.textContent = message;
  element.classList.remove("hidden");
}

function hideError(element) {
  element.classList.add("hidden");
  element.textContent = "";
}

/* ------------------------------------------------------------------ upload */

function handleFile(file) {
  if (!file) return;
  hideError(uploadError);
  dropzone.classList.add("busy");

  if (state.sourceUrl) URL.revokeObjectURL(state.sourceUrl);
  state.sourceUrl = URL.createObjectURL(file);
  state.file = file;
  const preview = new FontFace(`niv-source-${Date.now()}`, `url(${state.sourceUrl})`);
  preview
    .load()
    .then((loaded) => {
      document.fonts.add(loaded);
      $("font-preview").style.fontFamily = `"${loaded.family}", system-ui`;
    })
    .catch(() => {
      $("font-preview").style.fontFamily = "system-ui";
    });

  file
    .arrayBuffer()
    .then((buffer) => request("inspect", { buffer }, [buffer]))
    .then(({ info }) => {
      state.info = { ...info, filename: file.name };
      renderFontCard(state.info);
      buildAxisButtons();
      resetResult();
      settingsCard.classList.remove("is-locked");
      generateNote.textContent = "";
      scheduleEstimate();
    })
    .catch((error) => {
      state.info = null;
      fontCard.classList.add("hidden");
      settingsCard.classList.add("is-locked");
      resultCard.classList.add("is-locked");
      generateBtn.disabled = true;
      showError(uploadError, error.message);
      generateNote.textContent = "העלו פונט כדי להתחיל";
    })
    .finally(() => {
      dropzone.classList.remove("busy");
      fileInput.value = "";
    });
}

function renderFontCard(info) {
  $("font-name").textContent = info.name;
  const stats = [
    ["גליפים", info.glyphs],
    ["תווים ממופים", info.mapped],
    ["פשוטים", info.simple],
    ["UPM", info.unitsPerEm],
  ];
  $("font-stats").innerHTML = stats
    .map(([label, value]) => `<span class="chip">${label} <b>${nf.format(value)}</b></span>`)
    .join("");
  fontCard.classList.remove("hidden");

  const counts = {
    latin: info.presets.latin.simple,
    hebrew: info.presets.hebrew.simple,
    latin_hebrew: info.presets.latin_hebrew.simple,
    all: info.presets.all.simple,
  };
  document.querySelectorAll("[data-count]").forEach((el) => {
    const count = counts[el.dataset.count];
    el.textContent = count ? `${nf.format(count)} תווים` : "אין כיסוי";
  });
}

function buildAxisButtons() {
  const grid = $("axis-grid");
  grid.innerHTML = "";
  const axes = state.info ? state.info.axes : {};
  for (const [tag, meta] of Object.entries(axes)) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "axis-btn";
    button.dataset.axis = tag;
    button.setAttribute("aria-pressed", String(state.axes.has(tag)));
    button.innerHTML = `
      <span class="axis-tag">${tag}</span>
      <span class="axis-label">${meta.label}</span>
      <span class="axis-hint">${meta.hint}</span>`;
    button.addEventListener("click", () => toggleAxis(tag));
    grid.appendChild(button);
  }
  syncAxisButtons();
}

// One click: every axis the model supports, in a single file, and go.
allAxesBtn.addEventListener("click", () => {
  if (!state.info || state.generating) return;
  state.axes = new Set(Object.keys(state.info.axes));
  state.deltaLevel = 1;
  state.autoStart = true;
  syncAxisButtons();
  scheduleEstimate();
});

function toggleAxis(tag) {
  if (state.axes.has(tag)) {
    if (state.axes.size === 1) return;
    state.axes.delete(tag);
  } else {
    state.axes.add(tag);
  }
  state.deltaLevel = Math.min(state.deltaLevel, state.axes.size);
  syncAxisButtons();
  scheduleEstimate();
}

function syncAxisButtons() {
  document.querySelectorAll(".axis-btn").forEach((button) => {
    button.setAttribute("aria-pressed", String(state.axes.has(button.dataset.axis)));
  });
  document.querySelectorAll("#delta-level button").forEach((button) => {
    const level = Number(button.dataset.level);
    const enabled = level <= state.axes.size;
    button.disabled = !enabled;
    button.classList.toggle("active", enabled && level === state.deltaLevel);
  });
  allAxesBtn.disabled = !state.info || state.generating;
}

$("delta-level").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-level]");
  if (!button || button.disabled) return;
  state.deltaLevel = Number(button.dataset.level);
  syncAxisButtons();
  scheduleEstimate();
});

$("range-grid").addEventListener("click", (event) => {
  const button = event.target.closest(".range-chip");
  if (!button) return;
  state.preset = button.dataset.preset;
  document.querySelectorAll(".range-chip").forEach((chip) => chip.classList.toggle("active", chip === button));
  customRange.classList.toggle("hidden", state.preset !== "custom");
  if (state.preset === "custom") customRange.focus();
  scheduleEstimate();
});

customRange.addEventListener("input", () => {
  state.custom = customRange.value;
  scheduleEstimate();
});

/* ---------------------------------------------------------------- estimate */

let estimateTimer = null;

function scheduleEstimate() {
  if (!state.info) return;
  state.estimate = null;
  generateBtn.disabled = true;
  estimateText.textContent = "מחשב הערכה…";
  if (estimateTimer) clearTimeout(estimateTimer);
  estimateTimer = setTimeout(runEstimate, 220);
}

async function runEstimate() {
  if (!state.info) return;
  if (state.preset === "custom" && !state.custom.trim()) {
    state.autoStart = false;
    estimateText.textContent = "הזינו טווח Unicode, למשל 0x5D0-0x5EA";
    generateBtn.disabled = true;
    return;
  }
  try {
    const { value } = await request("estimate", {
      axes: [...state.axes],
      deltaLevel: state.deltaLevel,
      preset: state.preset,
      custom: state.custom,
    });
    state.estimate = value;
    if (!value.glyphs) {
      state.autoStart = false;
      estimateText.textContent = "לא נמצאו תווים מתאימים בטווח הזה בפונט";
      generateBtn.disabled = true;
      return;
    }
    const compositeNote = value.composite
      ? ` · ${nf.format(value.composite)} תווים מורכבים יישארו ללא וריאציה`
      : "";
    // ~0.05s per tuple on a typical laptop (single-threaded WASM).
    const seconds = value.tuples * 0.02;
    const timeNote = seconds > 2 ? ` · כ־${Math.round(seconds)} שנ׳` : "";
    estimateText.textContent = `${nf.format(value.glyphs)} תווים · ${nf.format(value.tuples)} וריאציות${timeNote}${compositeNote}`;
    generateBtn.disabled = false;
    if (state.autoStart) {
      state.autoStart = false;
      startGeneration();
    }
  } catch (error) {
    state.autoStart = false;
    estimateText.textContent = error.message;
    generateBtn.disabled = true;
  }
}

/* -------------------------------------------------------------- generation */

generateBtn.addEventListener("click", startGeneration);

async function startGeneration() {
  if (!state.info || !state.estimate || state.generating) return;
  resetResult();
  state.generating = true;
  state.startedAt = Date.now();
  generateBtn.disabled = true;
  syncAxisButtons();
  progressWrap.classList.remove("hidden");
  progressBar.style.width = "2%";
  progressMessage.textContent = state.modelReady ? "מתחיל לחשב…" : "טוען את המודל…";
  progressEta.textContent = "";

  const started = Date.now();
  try {
    const response = await request("generate", {
      axes: [...state.axes],
      deltaLevel: state.deltaLevel,
      preset: state.preset,
      custom: state.custom,
    });
    if (response.type === "cancelled") {
      progressWrap.classList.add("hidden");
      generateBtn.disabled = false;
      runEstimate();
      return;
    }
    progressBar.style.width = "100%";
    progressWrap.classList.add("hidden");
    showResult(response.bytes, response.stats, started);
    generateBtn.disabled = false;
  } catch (error) {
    progressWrap.classList.add("hidden");
    generateBtn.disabled = false;
    if (!/cancelled|בוטל/i.test(error.message)) {
      showError(jobError, error.message);
    } else {
      estimateText.textContent = "הפעולה בוטלה";
      runEstimate();
    }
  } finally {
    state.generating = false;
    syncAxisButtons();
  }
}

function updateProgress(message) {
  if (!state.generating) return;
  const fraction = Math.max(0.02, message.fraction || 0);
  progressBar.style.width = `${(fraction * 100).toFixed(1)}%`;
  progressMessage.textContent = `מחשב וריאציות… ${nf.format(message.tuples || 0)}`;
  const elapsed = (Date.now() - (state.startedAt || Date.now())) / 1000;
  if (fraction > 0.05) {
    const eta = (elapsed * (1 - fraction)) / fraction;
    progressEta.textContent = eta > 1 ? `נותרו ~${Math.round(eta)} שנ׳` : "";
  }
}

$("cancel-btn").addEventListener("click", () => {
  worker.postMessage({ type: "cancel" });
  progressMessage.textContent = "מבטל…";
});

/* ------------------------------------------------------------------ result */

function showResult(bytes, stats, started) {
  hideError(jobError);
  resultBody.classList.remove("hidden");
  resultCard.classList.remove("is-locked");

  if (state.fontUrl) URL.revokeObjectURL(state.fontUrl);
  state.fontUrl = URL.createObjectURL(new Blob([bytes], { type: "font/ttf" }));
  const family = `niv-${Date.now()}`;
  const face = new FontFace(family, `url(${state.fontUrl})`);
  face
    .load()
    .then((loaded) => {
      document.fonts.add(loaded);
      if (state.fontFace) document.fonts.delete(state.fontFace);
      state.fontFace = loaded;
      specimen.style.fontFamily = `"${family}", system-ui`;
      $("glyph-grid").style.fontFamily = `"${family}", system-ui`;
      applyVariation();
    })
    .catch(() => {
      showError(jobError, "הפונט נוצר אך לא ניתן להציג אותו בדפדפן. אפשר עדיין להוריד אותו.");
    });

  buildSliders(stats.axes);
  applyVariation();

  const base = (state.info && state.info.filename ? state.info.filename : "font").replace(/\.[^.]+$/, "");
  const download = $("download-btn");
  download.href = state.fontUrl;
  download.download = `${base}-variable.ttf`;

  const sample = /[\u0590-\u05EA]/.test(stats.chars || "") ? "אבגדה שלום Hello 123" : "The quick brown fox 123";
  $("sample-text").value = sample;
  specimen.textContent = sample;

  renderStats({
    ...stats,
    seconds: Math.max(0.1, (Date.now() - started) / 1000),
  });
  renderGlyphs(stats.chars || "");
  resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
}

function buildSliders(axes) {
  slidersEl.innerHTML = "";
  for (const tag of axes) {
    const meta = (state.info && state.info.axes[tag]) || { label: tag };
    const row = document.createElement("div");
    row.className = "slider-row";
    row.innerHTML = `
      <label>${meta.label} <span class="tag">${tag}</span></label>
      <input type="range" min="-1" max="1" step="0.05" value="0" data-tag="${tag}" />
      <span class="value" data-value="${tag}" dir="ltr">0.00</span>`;
    const input = row.querySelector("input");
    input.addEventListener("input", () => {
      row.querySelector(".value").textContent = Number(input.value).toFixed(2);
      applyVariation();
    });
    slidersEl.appendChild(row);
  }
}

function applyVariation() {
  const settings = [...slidersEl.querySelectorAll('input[type="range"]')]
    .map((input) => `"${input.dataset.tag}" ${Number(input.value).toFixed(2)}`)
    .join(", ");
  specimen.style.fontVariationSettings = settings;
  $("glyph-grid").style.fontVariationSettings = settings;
}

$("sample-text").addEventListener("input", (event) => {
  specimen.textContent = event.target.value || " ";
});

$("size-range").addEventListener("input", (event) => {
  specimen.style.fontSize = `${event.target.value}px`;
  $("size-value").textContent = event.target.value;
});

function renderStats(stats) {
  const rows = [
    ["תווים שנבחרו", nf.format(stats.codepoints)],
    ["גליפים עם וריאציה", nf.format(stats.glyphs)],
    ["וריאציות (tuples)", nf.format(stats.tuples)],
    ["זמן חישוב", `${stats.seconds.toFixed(1)} שנ׳`],
    ["צירים", stats.axes.join(", ")],
    ["רמת דלתא", String(stats.deltaLevel)],
    ["גודל קובץ", `${(stats.size / 1024).toFixed(0)} KB`],
  ];
  $("stats-grid").innerHTML = rows
    .map(([label, value]) => `<div class="stat"><span>${label}</span><b>${value}</b></div>`)
    .join("");
}

function renderGlyphs(chars) {
  const grid = $("glyph-grid");
  grid.innerHTML = "";
  const fragment = document.createDocumentFragment();
  for (const char of chars.slice(0, 400)) {
    if (char === "\u00A0") continue;
    const cell = document.createElement("span");
    cell.textContent = char === " " ? "␣" : char;
    fragment.appendChild(cell);
  }
  grid.appendChild(fragment);
}

/* ------------------------------------------------------------------- reset */

function resetResult() {
  state.generating = false;
  progressWrap.classList.add("hidden");
  resultBody.classList.add("hidden");
  hideError(jobError);
  resultCard.classList.add("is-locked");
  if (state.fontFace) {
    document.fonts.delete(state.fontFace);
    state.fontFace = null;
  }
}

$("reset-font").addEventListener("click", () => fileInput.click());

$("again-btn").addEventListener("click", () => {
  resetResult();
  generateBtn.disabled = false;
  settingsCard.scrollIntoView({ behavior: "smooth", block: "start" });
});

fileInput.addEventListener("change", (event) => handleFile(event.target.files[0]));

["dragenter", "dragover"].forEach((type) => {
  dropzone.addEventListener(type, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((type) => {
  dropzone.addEventListener(type, (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragover");
  });
});

dropzone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files && event.dataTransfer.files[0];
  handleFile(file);
});

setModelStatus("מכין את המודל…", "loading");
worker.postMessage({ type: "warmup" });
