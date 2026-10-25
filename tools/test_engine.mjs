/**
 * Runs the browser engine (docs/engine) under Node against the exported ONNX
 * model, so the generated font can be compared with the Python pipeline.
 *
 *   node tools/test_engine.mjs <font.ttf> <out.ttf> <axes> <preset> <deltaLevel>
 */
import fs from "node:fs";
import path from "node:path";
import ort from "onnxruntime-node";

const { parseFont } = await import("../docs/engine/ttf.js");
const { buildComboTable } = await import("../docs/engine/model.js");
const { generateVariableFont } = await import("../docs/engine/pipeline.js");

const fontPath = process.argv[2] || "samples/Heebo-Regular-static.ttf";
const outPath = process.argv[3] || "/tmp/js_out.ttf";
const axes = (process.argv[4] || "wght").split(",");
const preset = process.argv[5] || "latin_hebrew";
const deltaLevel = Number(process.argv[6] || 1);

const PRESETS = {
  latin: [[0x20, 0x7e]],
  hebrew: [[0x5d0, 0x5f4]],
  latin_hebrew: [
    [0x20, 0x7e],
    [0x5d0, 0x5f4],
  ],
};

const buffer = fs.readFileSync(fontPath);
const arrayBuffer = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
const font = parseFont(arrayBuffer);
const cfg = JSON.parse(fs.readFileSync("model/config.json", "utf8"));
const comboTable = buildComboTable(cfg.axes.length);

const session = await ort.InferenceSession.create(
  process.env.ONNX_MODEL || "docs/models/niv.onnx"
);
const run = async ({ points, padMask, positions, indices, weights, batch, pointsPerSample }) => {
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
  const results = await session.run(feeds);
  return results.deltas.data;
};

let codepoints;
if (preset === "all") codepoints = [...font.cmap.keys()];
else codepoints = PRESETS[preset].flatMap(([lo, hi]) => {
  const out = [];
  for (let cp = lo; cp <= hi; cp += 1) out.push(cp);
  return out;
});

const started = Date.now();
const result = await generateVariableFont({
  font,
  run,
  axes,
  codepoints,
  deltaLevel,
  axisIds: axes.map((tag) => cfg.axes.indexOf(tag)),
  comboTable,
  maxOrder: cfg.combinatorial_max_order,
  onProgress: (fraction, info) => {
    if (Math.round(fraction * 100) % 20 === 0) {
      console.log(`  ${(fraction * 100).toFixed(0)}% glyphs=${info.glyphs} tuples=${info.tuples}`);
    }
  },
});
fs.writeFileSync(outPath, Buffer.from(result.bytes));
console.log(
  `wrote ${outPath} (${(result.bytes.length / 1024).toFixed(0)} KB) glyphs=${result.glyphs} tuples=${result.tuples} in ${(
    (Date.now() - started) / 1000
  ).toFixed(1)}s`
);
