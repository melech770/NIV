/**
 * Web Worker: loads the exported NIV model with ONNX Runtime Web, runs inference
 * for every glyph and builds the variable font. No font data leaves the browser.
 */
import * as ort from "./vendor/ort.min.js";
import { parseFont } from "./engine/ttf.js";
import {
  buildComboTable,
  collectGlyphs,
  buildGlyphInput,
  conditioningTerms,
  permutations,
} from "./engine/model.js";
import { generateVariableFont } from "./engine/pipeline.js";
import { inspectFont } from "./engine/inspect.js";
import { parseRangeSpec } from "./engine/unicode.js";

let modelPromise = null;
let config = null;
let comboTable = null;
let parsedFont = null;
let cancelRequested = false;

let currentId = null;
const post = (payload, transfer) => {
  const id = payload && payload.id !== undefined ? payload.id : currentId;
  self.postMessage({ ...payload, id }, transfer || []);
};

async function fetchWithProgress(url, onProgress) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`טעינת המודל נכשלה (HTTP ${response.status})`);
  const total = Number(response.headers.get("content-length") || 0);
  if (!response.body || !total) {
    return new Uint8Array(await response.arrayBuffer());
  }
  const reader = response.body.getReader();
  const bytes = new Uint8Array(total);
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    bytes.set(value, loaded);
    loaded += value.length;
    onProgress(loaded / total);
  }
  return bytes;
}

function loadModel() {
  if (modelPromise) return modelPromise;
  modelPromise = (async () => {
    ort.env.wasm.wasmPaths = new URL("./vendor/", self.location.href).href;
    ort.env.wasm.numThreads = 1;
    ort.env.logLevel = "error";

    const base = new URL("./models/", self.location.href).href;
    config = await (await fetch(`${base}config.json`)).json();
    comboTable = buildComboTable(config.axes.length);

    let lastReport = 0;
    const bytes = await fetchWithProgress(`${base}niv.onnx`, (fraction) => {
      const now = Date.now();
      if (now - lastReport > 120 || fraction === 1) {
        lastReport = now;
        post({ type: "model-progress", fraction });
      }
    });

    const session = await ort.InferenceSession.create(bytes, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    post({ type: "model-ready" });
    return session;
  })();
  modelPromise.catch((error) => {
    modelPromise = null;
    post({ type: "model-error", message: error.message || String(error) });
  });
  return modelPromise;
}

function axisIdsFor(axes) {
  return axes.map((tag) => {
    const index = config.axes.indexOf(tag);
    if (index < 0) throw new Error(`המודל אינו תומך בציר ${tag}`);
    return index;
  });
}

function runSession(session) {
  return async ({ points, padMask, positions, indices, weights, batch, pointsPerSample }) => {
    const feeds = {
      points: new ort.Tensor("float32", points, [batch, pointsPerSample, 4]),
      pad_mask: new ort.Tensor("bool", padMask, [batch, pointsPerSample]),
      positions: new ort.Tensor("float32", positions, [batch, pointsPerSample]),
      combo_idx: new ort.Tensor(
        "int64",
        BigInt64Array.from(Array.from(indices, (value) => BigInt(value))),
        [indices.length]
      ),
      combo_w: new ort.Tensor("float32", weights, [weights.length]),
    };
    const output = await session.run(feeds);
    return output.deltas.data;
  };
}

function resolveCodepoints({ preset, custom }) {
  if (preset === "custom") return [...parseRangeSpec(custom)];
  if (preset === "all") return [...parsedFont.cmap.keys()];
  const ranges = {
    latin: [[0x20, 0x7e]],
    hebrew: [[0x5d0, 0x5f4]],
    latin_hebrew: [
      [0x20, 0x7e],
      [0x5d0, 0x5f4],
    ],
  }[preset];
  if (!ranges) throw new Error("בחירת תווים לא מוכרת");
  const out = [];
  for (const [lo, hi] of ranges) for (let cp = lo; cp <= hi; cp += 1) out.push(cp);
  return out;
}

function estimate({ axes, preset, custom, deltaLevel }) {
  const codepoints = resolveCodepoints({ preset, custom });
  const glyphs = collectGlyphs(parsedFont, codepoints).filter((entry) =>
    buildGlyphInput(parsedFont, entry.gid)
  );
  const perms = permutations(axes, deltaLevel).length;
  let composite = 0;
  const seen = new Set();
  for (const cp of codepoints) {
    const gid = parsedFont.cmap.get(cp);
    if (gid === undefined || seen.has(gid)) continue;
    seen.add(gid);
    if (!buildGlyphInput(parsedFont, gid)) composite += 1;
  }
  return { glyphs: glyphs.length, composite, tuples: glyphs.length * perms };
}

self.onmessage = async (event) => {
  const message = event.data || {};
  currentId = message.id ?? null;
  try {
    if (message.type === "warmup") {
      loadModel();
      return;
    }

    if (message.type === "inspect") {
      const font = parseFont(message.buffer);
      parsedFont = font;
      post({ type: "inspected", info: inspectFont(font) });
      return;
    }

    if (message.type === "estimate") {
      if (!parsedFont) throw new Error("העלה פונט קודם");
      post({ type: "estimate", value: estimate(message) });
      return;
    }

    if (message.type === "generate") {
      if (!parsedFont) throw new Error("העלה פונט קודם");
      const generateId = currentId;
      const session = await loadModel();
      cancelRequested = false;
      const codepoints = resolveCodepoints(message);
      const axes = message.axes;
      let lastReport = 0;
      const started = Date.now();

      const result = await generateVariableFont({
        font: parsedFont,
        run: runSession(session),
        axes,
        codepoints,
        deltaLevel: message.deltaLevel,
        axisIds: axisIdsFor(axes),
        comboTable,
        maxOrder: config.combinatorial_max_order,
        predictionType: config.prediction_type,
        batchSize: 6,
        isCancelled: () => cancelRequested,
        onProgress: (fraction, info) => {
          const now = Date.now();
          if (now - lastReport > 90 || fraction === 1) {
            lastReport = now;
            post({ type: "progress", fraction, id: generateId, ...info });
          }
        },
      });

      const chars = codepoints
        .filter((cp) => cp >= 0x20 && cp <= 0x2fff)
        .sort((a, b) => a - b)
        .map((cp) => String.fromCodePoint(cp))
        .join("");
      const bytes = result.bytes;
      post(
        {
          type: "done",
          id: generateId,
          bytes: bytes.buffer,
          stats: {
            glyphs: result.glyphs,
            tuples: result.tuples,
            codepoints: codepoints.length,
            axes,
            deltaLevel: message.deltaLevel,
            seconds: (Date.now() - started) / 1000,
            size: bytes.length,
            chars,
          },
        },
        [bytes.buffer]
      );
      return;
    }

    if (message.type === "cancel") {
      cancelRequested = true;
      return;
    }
  } catch (error) {
    if (cancelRequested && error && error.message === "cancelled") {
      post({ type: "cancelled", id: message.id ?? null });
      cancelRequested = false;
      return;
    }
    const message_text =
      error && error.message === "already-variable"
        ? "הפונט הזה כבר פונט משתנה (יש בו צירי fvar)."
        : error && error.message === "no-glyf"
        ? "הפונט אינו מבוסס outlines מסוג TrueType (glyf); פונטי CFF/OTF אינם נתמכים."
        : error && error.message === "unsupported-outline-format"
        ? "זה לא נראה כמו קובץ פונט TrueType תקין."
        : (error && error.message) || String(error);
    post({ type: "error", message: message_text, context: message.type, id: message.id ?? null });
  }
};
