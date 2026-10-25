/** Runs the ONNX model over a font's glyphs and assembles the variable font. */
import {
  buildGlyphInput,
  collectGlyphs,
  conditioningTerms,
  permutations,
} from "./model.js";
import { buildVariableFont } from "./variable.js";

function axisWeight(min, peak, max, value) {
  if (value <= min || value >= max) return Math.abs(value - peak) < 1e-9 ? 1 : 0;
  if (Math.abs(value - peak) < 1e-9) return 1;
  if (value < peak) {
    return Math.abs(peak - min) < 1e-9 ? 0 : (value - min) / (peak - min);
  }
  return Math.abs(max - peak) < 1e-9 ? 0 : (max - value) / (max - peak);
}

function tupleWeight(peak, values) {
  let weight = 1;
  for (const [tag, value] of Object.entries(peak)) {
    weight *= axisWeight(Math.min(value, 0), value, Math.max(value, 0), values[tag] ?? 0);
    if (weight === 0) break;
  }
  return weight;
}

function batchInputs(entries) {
  const points = entries[entries.length - 1].input.pointCount;
  const batch = entries.length;
  const tensor = new Float32Array(batch * points * 4);
  const pad = new Uint8Array(batch * points);
  const positions = new Float32Array(batch * points);
  entries.forEach((entry, row) => {
    tensor.set(entry.input.data, row * points * 4);
    for (let i = 0; i < points; i += 1) {
      positions[row * points + i] = i;
      pad[row * points + i] = i >= entry.input.pointCount ? 1 : 0;
    }
  });
  return { tensor, pad, positions, points };
}

/**
 * @param {object} options
 * @param {object} options.font parsed font (see ttf.js)
 * @param {(inputs: object) => Promise<Float32Array>} options.run ONNX runner
 * @param {string[]} options.axes axis tags in the order used for fvar
 * @param {number[]} options.codepoints codepoints to cover
 * @param {number} options.deltaLevel
 * @param {number[]} options.axisIds model axis id per requested axis
 * @param {Map} options.comboTable
 * @param {number} options.maxOrder combinatorial_max_order from config
 */
export async function generateVariableFont(options) {
  const {
    font,
    run,
    axes,
    codepoints,
    deltaLevel,
    axisIds,
    comboTable,
    maxOrder,
    onProgress,
    isCancelled,
    batchSize = 6,
    predictionType = "total",
  } = options;

  const glyphs = collectGlyphs(font, codepoints)
    .map((entry) => ({ ...entry, input: buildGlyphInput(font, entry.gid) }))
    .filter((entry) => entry.input);

  const perms = permutations(axes, deltaLevel).map((permutation) => ({
    ...permutation,
    terms: conditioningTerms(permutation, axes, axisIds, comboTable, maxOrder),
    peak: Object.fromEntries(
      permutation.nonZero.map((index) => [axes[index], permutation.values[index]])
    ),
  }));

  const variations = new Map();
  for (const entry of glyphs) variations.set(entry.gid, []);
  if (!glyphs.length || !perms.length) {
    return { bytes: buildVariableFont(font, axes, variations), glyphs: 0, tuples: 0 };
  }

  const sorted = [...glyphs].sort(
    (a, b) => a.input.pointCount - b.input.pointCount || a.cp - b.cp
  );
  const chunks = [];
  for (let i = 0; i < sorted.length; i += batchSize) chunks.push(sorted.slice(i, i + batchSize));

  const total = chunks.length * perms.length;
  let completed = 0;
  let tuples = 0;

  for (const perm of perms) {
    for (const chunk of chunks) {
      if (isCancelled && isCancelled()) throw new Error("cancelled");
      const { tensor, pad, positions, points } = batchInputs(chunk);
      const output = await run({
        points: tensor,
        padMask: pad,
        positions,
        indices: Int32Array.from(perm.terms.indices),
        weights: Float32Array.from(perm.terms.weights),
        batch: chunk.length,
        pointsPerSample: points,
      });
      const scale = font.unitsPerEm / 2;
      const axisValues = Object.fromEntries(
        axes.map((tag, index) => [tag, perm.values[index]])
      );
      chunk.forEach((entry, row) => {
        const pointCount = entry.input.pointCount;
        const existing = variations.get(entry.gid);
        const deltas = new Array(pointCount);
        for (let i = 0; i < pointCount; i += 1) {
          const base = (row * points + i) * 2;
          deltas[i] = [output[base] * scale, output[base + 1] * scale];
        }

        // 'total' prediction: combined tuples store only what single-axis tuples
        // do not already explain, matching code/ttf_to_var.py.
        if (predictionType === "total" && existing.length) {
          const accumulated = new Array(pointCount).fill(null).map(() => [0, 0]);
          for (const tuple of existing) {
            const weight = tupleWeight(tuple.peak, axisValues);
            if (weight <= 0) continue;
            for (let i = 0; i < pointCount; i += 1) {
              accumulated[i][0] += weight * tuple.deltas[i][0];
              accumulated[i][1] += weight * tuple.deltas[i][1];
            }
          }
          for (let i = 0; i < pointCount; i += 1) {
            deltas[i][0] = Math.round(deltas[i][0] - accumulated[i][0]);
            deltas[i][1] = Math.round(deltas[i][1] - accumulated[i][1]);
          }
        } else {
          for (let i = 0; i < pointCount; i += 1) {
            deltas[i][0] = Math.round(deltas[i][0]);
            deltas[i][1] = Math.round(deltas[i][1]);
          }
        }
        existing.push({ peak: perm.peak, deltas, pointCount });
        tuples += 1;
      });
      completed += 1;
      if (onProgress) onProgress(completed / total, { tuples, glyphs: glyphs.length });
      // Yield so the worker can handle cancellation messages.
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
  }

  const bytes = buildVariableFont(font, axes, variations);
  return { bytes, glyphs: glyphs.length, tuples };
}

export function estimateWork(font, codepoints, axesCount, deltaLevel) {
  const glyphs = collectGlyphs(font, codepoints).filter((entry) => buildGlyphInput(font, entry.gid));
  let perms = 0;
  const count = axesCount;
  const total = 3 ** count;
  for (let code = 0; code < total; code += 1) {
    let values = [];
    for (let i = 0; i < count; i += 1) {
      values.push((Math.floor(code / 3 ** (count - 1 - i)) % 3) - 1);
    }
    const nonZero = values.filter((value) => value !== 0).length;
    if (nonZero > 0 && nonZero <= deltaLevel) perms += 1;
  }
  return { glyphs: glyphs.length, tuples: glyphs.length * perms };
}
