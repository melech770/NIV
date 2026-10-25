const state = {
  token: null,
  info: null,
  axes: new Set(["wght"]),
  deltaLevel: 1,
  preset: "latin",
  custom: "",
  jobId: null,
  pollTimer: null,
  estimate: null,
  fontFamily: null,
  result: null,
};

const $ = (id) => document.getElementById(id);

const uploadCard = $("upload-card");
const settingsCard = $("settings-card");
const resultCard = $("result-card");
const dropzone = $("dropzone");
const fileInput = $("file-input");
const fontCard = $("font-card");
const uploadError = $("upload-error");
const generateBtn = $("generate-btn");
const generateNote = $("generate-note");
const estimateText = $("estimate-text");
const customRange = $("custom-range");
const progressWrap = $("progress-wrap");
const progressBar = $("progress-bar");
const progressMessage = $("progress-message");
const progressEta = $("progress-eta");
const jobError = $("job-error");
const resultBody = $("result-body");
const specimen = $("specimen");
const slidersEl = $("sliders");

/* ------------------------------------------------------------------ utils */

const nf = new Intl.NumberFormat("he-IL");

function formatDuration(seconds) {
  if (!isFinite(seconds) || seconds <= 0) return "פחות משנייה";
  if (seconds < 60) return `${Math.max(1, Math.round(seconds))} שניות`;
  return `${(seconds / 60).toFixed(1)} דקות`;
}

function showError(el, message) {
  el.textContent = message;
  el.classList.remove("hidden");
}

function hideError(el) {
  el.classList.add("hidden");
  el.textContent = "";
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  let payload = null;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }
  if (!response.ok) {
    const detail = payload && payload.detail ? payload.detail : `שגיאת שרת (${response.status})`;
    throw new Error(typeof detail === "string" ? detail : "בקשה לא תקינה");
  }
  return payload;
}

/* ---------------------------------------------------------------- upload */

async function handleFile(file) {
  if (!file) return;
  hideError(uploadError);
  dropzone.classList.add("busy");

  const body = new FormData();
  body.append("file", file);
  try {
    const info = await api("/api/inspect", { method: "POST", body });
    state.info = info;
    state.token = info.token;
    state.result = null;
    renderFontCard(info);
    buildAxisButtons();
    resetResult();
    unlockSettings();
    loadSourcePreview(info.token);
    scheduleEstimate();
  } catch (error) {
    state.info = null;
    state.token = null;
    fontCard.classList.add("hidden");
    lockSettings();
    showError(uploadError, error.message);
  } finally {
    dropzone.classList.remove("busy");
    fileInput.value = "";
  }
}

async function loadSourcePreview(token) {
  try {
    const face = new FontFace(`niv-source-${token}`, `url(/api/source/${token}.ttf)`);
    const loaded = await face.load();
    document.fonts.add(loaded);
    $("font-preview").style.fontFamily = `"niv-source-${token}", system-ui`;
  } catch {
    $("font-preview").style.fontFamily = "system-ui";
  }
}

function renderFontCard(info) {
  $("font-name").textContent = info.name;
  const stats = [
    ["גליפים", info.glyphs],
    ["תווים ממופים", info.mapped],
    ["פשוטים", info.simple],
    ["UPM", info.units_per_em],
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
    const n = counts[el.dataset.count];
    el.textContent = n ? `${nf.format(n)} תווים` : "אין כיסוי";
  });
}

function lockSettings() {
  settingsCard.classList.add("is-locked");
  resultCard.classList.add("is-locked");
  generateBtn.disabled = true;
  generateNote.textContent = "העלה פונט כדי להתחיל";
}

function unlockSettings() {
  settingsCard.classList.remove("is-locked");
  generateNote.textContent = "";
}

function resetResult() {
  if (state.pollTimer) clearInterval(state.pollTimer);
  state.pollTimer = null;
  state.jobId = null;
  progressWrap.classList.add("hidden");
  resultBody.classList.add("hidden");
  hideError(jobError);
  resultCard.classList.add("is-locked");
  if (state.fontFamily) {
    document.fonts.delete(state.fontFamily);
    state.fontFamily = null;
  }
}

/* --------------------------------------------------------------- settings */

function buildAxisButtons() {
  const grid = $("axis-grid");
  grid.innerHTML = "";
  const axes = state.info ? state.info.axes : {};
  for (const [tag, meta] of Object.entries(axes)) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "axis-btn";
    btn.dataset.axis = tag;
    btn.setAttribute("aria-pressed", String(state.axes.has(tag)));
    btn.innerHTML = `
      <span class="axis-tag">${tag}</span>
      <span class="axis-label">${meta.label}</span>
      <span class="axis-hint">${meta.hint}</span>`;
    btn.addEventListener("click", () => toggleAxis(tag));
    grid.appendChild(btn);
  }
  syncAxisButtons();
}

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
  document.querySelectorAll(".axis-btn").forEach((btn) => {
    btn.setAttribute("aria-pressed", String(state.axes.has(btn.dataset.axis)));
  });
  document.querySelectorAll("#delta-level button").forEach((btn) => {
    const level = Number(btn.dataset.level);
    const enabled = level <= state.axes.size;
    btn.disabled = !enabled;
    btn.classList.toggle("active", enabled && level === state.deltaLevel);
  });
}

$("delta-level").addEventListener("click", (event) => {
  const btn = event.target.closest("button[data-level]");
  if (!btn || btn.disabled) return;
  state.deltaLevel = Number(btn.dataset.level);
  syncAxisButtons();
  scheduleEstimate();
});

$("range-grid").addEventListener("click", (event) => {
  const btn = event.target.closest(".range-chip");
  if (!btn) return;
  state.preset = btn.dataset.preset;
  document.querySelectorAll(".range-chip").forEach((chip) => chip.classList.toggle("active", chip === btn));
  customRange.classList.toggle("hidden", state.preset !== "custom");
  if (state.preset === "custom") customRange.focus();
  scheduleEstimate();
});

customRange.addEventListener("input", () => {
  state.custom = customRange.value;
  scheduleEstimate();
});

let estimateTimer = null;

function scheduleEstimate() {
  if (!state.token) return;
  state.estimate = null;
  generateBtn.disabled = true;
  estimateText.textContent = "מחשב הערכה…";
  if (estimateTimer) clearTimeout(estimateTimer);
  estimateTimer = setTimeout(runEstimate, 260);
}

async function runEstimate() {
  if (!state.token) return;
  if (state.preset === "custom" && !state.custom.trim()) {
    estimateText.textContent = "הזן טווח Unicode, למשל 0x5D0-0x5EA";
    generateBtn.disabled = true;
    return;
  }
  try {
    const data = await api("/api/estimate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        token: state.token,
        axes: [...state.axes],
        delta_level: state.deltaLevel,
        preset: state.preset,
        custom: state.custom,
      }),
    });
    state.estimate = data;
    if (!data.glyphs) {
      estimateText.textContent = "לא נמצאו תווים מתאימים בטווח הזה בפונט";
      generateBtn.disabled = true;
      return;
    }
    const compositeNote = data.composite ? ` · ${nf.format(data.composite)} תווים מורכבים יישארו ללא וריאציה` : "";
    estimateText.textContent = `${nf.format(data.glyphs)} תווים · ${nf.format(data.tuples)} וריאציות · כ־${formatDuration(
      data.seconds
    )}${compositeNote}`;
    generateBtn.disabled = false;
  } catch (error) {
    estimateText.textContent = error.message;
    generateBtn.disabled = true;
  }
}

/* ------------------------------------------------------------- generation */

generateBtn.addEventListener("click", startGeneration);

async function startGeneration() {
  if (!state.token || !state.estimate) return;
  resetResult();
  generateBtn.disabled = true;
  progressWrap.classList.remove("hidden");
  progressBar.style.width = "2%";
  progressMessage.textContent = "שולח את הפונט למודל…";
  progressEta.textContent = "";

  try {
    const job = await api("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        token: state.token,
        axes: [...state.axes],
        delta_level: state.deltaLevel,
        preset: state.preset,
        custom: state.custom,
      }),
    });
    state.jobId = job.id;
    resultCard.classList.remove("is-locked");
    pollJob(job.id);
    state.pollTimer = setInterval(() => pollJob(job.id), 600);
  } catch (error) {
    progressWrap.classList.add("hidden");
    generateBtn.disabled = false;
    showError(jobError, error.message);
  }
}

async function pollJob(jobId) {
  try {
    const job = await api(`/api/jobs/${jobId}`);
    progressBar.style.width = `${Math.max(2, job.progress * 100).toFixed(1)}%`;
    if (job.state === "queued") {
      progressMessage.textContent = "בתור לעיבוד…";
    } else {
      progressMessage.textContent = job.message || "מעבד…";
    }
    if (job.eta) progressEta.textContent = `נותרו ~${formatDuration(job.eta)}`;

    if (job.state === "done") {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
      progressBar.style.width = "100%";
      progressEta.textContent = "";
      progressMessage.textContent = "הושלם";
      setTimeout(() => progressWrap.classList.add("hidden"), 450);
      generateBtn.disabled = false;
      showResult(job);
    } else if (job.state === "error" || job.state === "cancelled") {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
      progressWrap.classList.add("hidden");
      generateBtn.disabled = false;
      if (job.state === "error") {
        showError(jobError, job.error || "העיבוד נכשל.");
      } else {
        estimateText.textContent = "הפעולה בוטלה";
        runEstimate();
      }
    }
  } catch (error) {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
    showError(jobError, error.message);
  }
}

$("cancel-btn").addEventListener("click", async () => {
  if (!state.jobId) return;
  try {
    await api(`/api/jobs/${state.jobId}/cancel`, { method: "POST" });
    progressMessage.textContent = "מבטל…";
  } catch (error) {
    showError(jobError, error.message);
  }
});

/* ----------------------------------------------------------------- result */

async function showResult(job) {
  const result = job.result;
  state.result = result;
  hideError(jobError);
  resultBody.classList.remove("hidden");
  resultCard.classList.remove("is-locked");

  const family = `niv-${job.id}`;
  try {
    const face = new FontFace(family, `url(${job.fontUrl})`);
    const loaded = await face.load();
    document.fonts.add(loaded);
    if (state.fontFamily) document.fonts.delete(state.fontFamily);
    state.fontFamily = loaded;
    specimen.style.fontFamily = `"${family}", system-ui`;
    $("glyph-grid").style.fontFamily = `"${family}", system-ui`;
  } catch (error) {
    showError(jobError, "הפונט נוצר אך לא ניתן להציג אותו בדפדפן. אפשר עדיין להוריד אותו.");
  }

  buildSliders(result.axes);
  applyVariation();

  $("download-btn").href = job.downloadUrl;
  $("sample-text").value = defaultSample(result);
  specimen.textContent = $("sample-text").value;
  renderStats(result);
  renderGlyphs(result);
  resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
}

function defaultSample(result) {
  const chars = result.chars || "";
  if (/[\u0590-\u05EA]/.test(chars)) return "אבגדה שלום Hello 123";
  return "The quick brown fox 123";
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

function renderStats(result) {
  const rows = [
    ["תווים שעובדו", nf.format(result.codepoints)],
    ["גליפים עם וריאציה", nf.format(result.glyphs)],
    ["וריאציות (tuples)", nf.format(result.tuples)],
    ["דילוג על מורכבים", nf.format(result.skipped_composite)],
    ["זמן ריצה", `${result.seconds} שנ׳`],
    ["צירים", result.axes.join(", ")],
    ["רמת דלתא", String(result.delta_level)],
    ["גודל קובץ", result.bytes ? `${(result.bytes / 1024).toFixed(0)} KB` : "—"],
  ];
  $("stats-grid").innerHTML = rows
    .map(([label, value]) => `<div class="stat"><span>${label}</span><b>${value}</b></div>`)
    .join("");
}

function renderGlyphs(result) {
  const grid = $("glyph-grid");
  grid.innerHTML = "";
  const chars = (result.chars || "").slice(0, 400);
  const fragment = document.createDocumentFragment();
  for (const char of chars) {
    if (char === "\u00A0") continue;
    const cell = document.createElement("span");
    cell.textContent = char === " " ? "␣" : char;
    fragment.appendChild(cell);
  }
  grid.appendChild(fragment);
}

/* ------------------------------------------------------------------ reset */

$("reset-font").addEventListener("click", () => {
  state.token = null;
  state.info = null;
  state.result = null;
  fontCard.classList.add("hidden");
  resetResult();
  lockSettings();
  fileInput.click();
});

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

lockSettings();
