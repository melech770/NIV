/**
 * End-to-end check of the exact browser runtime: ONNX Runtime **Web** (WASM,
 * single thread) + docs/engine, producing a font to compare with the Python
 * pipeline.
 *
 *   node tools/test_web_runtime.mjs [font.ttf] [out.ttf] [axes] [preset] [delta]
 */
import fs from "node:fs";
import path from "node:path";
import * as ort from "../docs/vendor/ort.min.js";
import { parseFont } from "../docs/engine/ttf.js";
import { buildComboTable } from "../docs/engine/model.js";
import { generateVariableFont } from "../docs/engine/pipeline.js";
import { presetCodepoints } from "../docs/engine/inspect.js";
import { parseRangeSpec } from "../docs/engine/unicode.js";

const fontPath = process.argv[2] || "samples/Heebo-Regular-static.ttf";
const outPath = process.argv[3] || "/tmp/wasm_out.ttf";
const axes = (process.argv[4] || "wght").split(",");
const preset = process.argv[5] || "latin_hebrew";
const deltaLevel = Number(process.argv[6] || 1);

ort.env.wasm.wasmPaths = `file://${path.resolve("docs/vendor")}/`;
ort.env.wasm.wasmBinary = fs.readFileSync("docs/vendor/ort-wasm-simd-threaded.wasm");
ort.env.wasm.numThreads = 1;
ort.env.logLevel = "error";

const buffer = fs.readFileSync(fontPath);
const font = parseFont(buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength));
const config = JSON.parse(fs.readFileSync("docs/models/config.json", "utf8"));
const comboTable = buildComboTable(config.axes.length);

const session = await ort.InferenceSession.create(fs.readFileSync("docs/models/niv.onnx"), {
  executionProviders: ["wasm"],
  graphOptimizationLevel: "all",
});

const run = async ({ points, padMask, positions, indices, weights, batch, pointsPerSample }) => {
  const output = await session.run({
    points: new ort.Tensor("float32", points, [batch, pointsPerSample, 4]),
    pad_mask: new ort.Tensor("bool", padMask, [batch, pointsPerSample]),
    positions: new ort.Tensor("float32", positions, [batch, pointsPerSample]),
    combo_idx: new ort.Tensor("int64", BigInt64Array.from(Array.from(indices, (v) => BigInt(v))), [indices.length]),
    combo_w: new ort.Tensor("float32", weights, [weights.length]),
  });
  return output.deltas.data;
};

const codepoints =
  preset === "custom"
    ? [...parseRangeSpec(process.env.RANGE || "0x20-0x7E")]
    : preset === "all"
    ? [...font.cmap.keys()]
    : presetCodepoints(preset);

const started = Date.now();
const result = await generateVariableFont({
  font,
  run,
  axes,
  codepoints,
  deltaLevel,
  axisIds: axes.map((tag) => config.axes.indexOf(tag)),
  comboTable,
  maxOrder: config.combinatorial_max_order,
  predictionType: config.prediction_type,
  batchSize: Number(process.env.BATCH || 6),
});
fs.writeFileSync(outPath, Buffer.from(result.bytes));
console.log(
  `wasm wrote ${outPath} glyphs=${result.glyphs} tuples=${result.tuples} in ${(
    (Date.now() - started) / 1000
  ).toFixed(1)}s`
);
