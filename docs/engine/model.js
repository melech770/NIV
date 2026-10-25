/** Axis permutations, ONNX conditioning terms and glyph input tensors. */
import { contourInfo, readGlyph } from "./ttf.js";

/**
 * combo key -> index, mirroring AxisIdAndValueEmbedder.combo_to_idx.
 * A key lists (axisId, sign) pairs sorted by axis id.
 */
export function buildComboTable(numAxes) {
  const table = new Map();
  let index = 0;
  const combinations = [];
  const generate = (start, remaining, current) => {
    if (remaining === 0) {
      combinations.push(current.slice());
      return;
    }
    for (let axisId = start; axisId < numAxes; axisId += 1) {
      current.push(axisId);
      generate(axisId + 1, remaining - 1, current);
      current.pop();
    }
  };
  for (let depth = 1; depth <= numAxes; depth += 1) {
    combinations.length = 0;
    generate(0, depth, []);
    for (const combo of combinations) {
      for (let signs = 0; signs < 1 << depth; signs += 1) {
        const key = combo.map((axisId, i) => [axisId, (signs >> i) & 1 ? 1 : -1]);
        table.set(JSON.stringify(key), index);
        index += 1;
      }
    }
  }
  return table;
}

export function permutations(axisTags, deltaLevel) {
  const results = [];
  const count = axisTags.length;
  const total = 3 ** count;
  for (let code = 0; code < total; code += 1) {
    const values = [];
    for (let i = 0; i < count; i += 1) {
      values.push((Math.floor(code / 3 ** (count - 1 - i)) % 3) - 1);
    }
    const nonZero = values.map((value, index) => (value !== 0 ? index : -1)).filter((i) => i >= 0);
    if (nonZero.length === 0 || nonZero.length > deltaLevel) continue;
    results.push({ values, nonZero });
  }
  results.sort((a, b) => a.nonZero.length - b.nonZero.length);
  return results;
}

/** Conditioning terms (combo index + weight) for one axis tuple. */
export function conditioningTerms(permutation, axisTags, axisIds, comboTable, maxOrder) {
  const active = permutation.nonZero
    .map((axisIndex) => ({ axisId: axisIds[axisIndex], value: permutation.values[axisIndex] }))
    .sort((a, b) => a.axisId - b.axisId);
  const indices = [];
  const weights = [];
  const walk = (start, remaining, key, weight) => {
    if (remaining === 0) {
      indices.push(comboTable.get(JSON.stringify(key)));
      weights.push(weight);
      return;
    }
    for (let position = start; position < active.length; position += 1) {
      const { axisId, value } = active[position];
      walk(position + 1, remaining - 1, key.concat([[axisId, value >= 0 ? 1 : -1]]), weight * Math.abs(value));
    }
  };
  for (let depth = 1; depth <= Math.min(active.length, maxOrder); depth += 1) walk(0, depth, [], 1);
  return { indices, weights };
}

/**
 * Feature tensor for one simple glyph: [x, y, onCurve, positionInContour] for each
 * outline point, followed by four phantom points. Coordinates are normalized to -1..1.
 */
export function buildGlyphInput(font, gid) {
  const outline = readGlyph(font, gid);
  if (!outline) return null;
  const upm = font.unitsPerEm;
  const count = outline.points.length;
  const total = count + 4;
  const { pos } = contourInfo(outline.points, outline.ends);
  const data = new Float32Array(total * 4);
  for (let i = 0; i < count; i += 1) {
    const point = outline.points[i];
    data[i * 4] = (point.x * 2) / upm - 1;
    data[i * 4 + 1] = (point.y * 2) / upm - 1;
    data[i * 4 + 2] = point.onCurve ? 1 : 0;
    data[i * 4 + 3] = pos[i];
  }
  for (let i = count; i < total; i += 1) {
    data[i * 4 + 2] = -1;
  }
  return { data, pointCount: total, outline };
}

export function collectGlyphs(font, codepoints) {
  const seen = new Set();
  const glyphs = [];
  for (const cp of codepoints) {
    const gid = font.cmap.get(cp);
    if (gid === undefined || seen.has(gid)) continue;
    seen.add(gid);
    glyphs.push({ cp, gid });
  }
  return glyphs;
}
