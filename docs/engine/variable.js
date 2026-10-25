/** Builds the fvar/avar/gvar tables that turn a static font into a variable one. */
import { addNameRecords, buildSfnt, f2dot14 } from "./ttf.js";

function fixed(value) {
  return Math.round(value * 65536);
}

function buildFvar(axes, nameIds) {
  const bytes = new Uint8Array(16 + axes.length * 20);
  const view = new DataView(bytes.buffer);
  view.setUint16(0, 1);
  view.setUint16(2, 0);
  view.setUint16(4, 16);
  view.setUint16(6, 2); // reserved
  view.setUint16(8, axes.length);
  view.setUint16(10, 20);
  view.setUint16(12, 0); // instanceCount
  view.setUint16(14, 4 + axes.length * 4); // instanceSize
  axes.forEach((tag, index) => {
    const base = 16 + index * 20;
    for (let i = 0; i < 4; i += 1) view.setUint8(base + i, tag.charCodeAt(i));
    view.setInt32(base + 4, fixed(-1));
    view.setInt32(base + 8, fixed(0));
    view.setInt32(base + 12, fixed(1));
    view.setUint16(base + 16, 0);
    view.setUint16(base + 18, nameIds.get(tag));
  });
  return bytes;
}

function buildAvar(axes) {
  const bytes = new Uint8Array(8 + axes.length * 14);
  const view = new DataView(bytes.buffer);
  view.setUint16(0, 1);
  view.setUint16(2, 0);
  view.setUint16(4, 0); // reserved
  view.setUint16(6, axes.length);
  axes.forEach((_, index) => {
    const base = 8 + index * 14;
    view.setUint16(base, 3);
    const segments = [
      [-1, -1],
      [0, 0],
      [1, 1],
    ];
    segments.forEach(([from, to], segmentIndex) => {
      view.setInt16(base + 2 + segmentIndex * 4, f2dot14(from));
      view.setInt16(base + 4 + segmentIndex * 4, f2dot14(to));
    });
  });
  return bytes;
}

function packDeltas(values) {
  const out = [];
  let index = 0;
  while (index < values.length) {
    if (values[index] === 0) {
      let run = 0;
      while (run < 64 && index + run < values.length && values[index + run] === 0) run += 1;
      out.push(0x80 | (run - 1));
      index += run;
      continue;
    }
    let run = 0;
    while (
      run < 64 &&
      index + run < values.length &&
      values[index + run] !== 0 &&
      values[index + run] >= -32768 &&
      values[index + run] <= 32767
    ) {
      run += 1;
    }
    out.push(0x40 | (run - 1));
    for (let i = 0; i < run; i += 1) {
      const value = values[index + i];
      out.push((value >> 8) & 0xff, value & 0xff);
    }
    index += run;
  }
  return Uint8Array.from(out);
}

function serializeGlyphVariations(axes, tuples, pointCount) {
  const headers = [];
  const dataBlocks = [];
  for (const tuple of tuples) {
    const xValues = new Int16Array(pointCount);
    const yValues = new Int16Array(pointCount);
    for (let i = 0; i < pointCount; i += 1) {
      xValues[i] = tuple.deltas[i][0];
      yValues[i] = tuple.deltas[i][1];
    }
    const deltaBytes = concat(packDeltas(xValues), packDeltas(yValues));
    const header = new Uint8Array(4 + axes.length * 2);
    const view = new DataView(header.buffer);
    view.setUint16(0, deltaBytes.length);
    view.setUint16(2, 0x8000); // embedded peak tuple, no intermediates
    axes.forEach((tag, index) => {
      view.setInt16(4 + index * 2, f2dot14(tuple.peak[tag] ?? 0));
    });
    headers.push(header);
    dataBlocks.push(deltaBytes);
  }

  const headerSize = headers.reduce((sum, header) => sum + header.length, 0);
  const dataSize = dataBlocks.reduce((sum, block) => sum + block.length, 0);
  // 0x8000: point numbers are shared by all tuples.  The shared block below is a
  // single 0x00, meaning "all points" (outline points + four phantom points).
  const total = 4 + headerSize + 1 + dataSize;
  const out = new Uint8Array(total);
  const view = new DataView(out.buffer);
  view.setUint16(0, tuples.length | 0x8000);
  view.setUint16(2, 4 + headerSize);
  let offset = 4;
  for (const header of headers) {
    out.set(header, offset);
    offset += header.length;
  }
  out[offset] = 0; // shared point numbers: count 0 = all points
  offset += 1;
  for (const block of dataBlocks) {
    out.set(block, offset);
    offset += block.length;
  }
  return out;
}

function concat(a, b) {
  const out = new Uint8Array(a.length + b.length);
  out.set(a, 0);
  out.set(b, a.length);
  return out;
}

function buildGvar(numGlyphs, axes, variationsByGlyph) {
  const blocks = new Array(numGlyphs).fill(null);
  let dataLength = 0;
  for (const [gid, tuples] of variationsByGlyph) {
    if (!tuples.length) continue;
    const pointCount = tuples[0].pointCount;
    const block = serializeGlyphVariations(axes, tuples, pointCount);
    const padded = block.length % 2 === 0 ? block : concat(block, new Uint8Array(1));
    blocks[gid] = padded;
    dataLength += padded.length;
  }

  const useLong = dataLength + numGlyphs * 4 > 0xffff;
  const offsets = new Uint32Array(numGlyphs + 1);
  let running = 0;
  for (let gid = 0; gid < numGlyphs; gid += 1) {
    offsets[gid] = running;
    running += blocks[gid] ? blocks[gid].length : 0;
  }
  offsets[numGlyphs] = running;

  const offsetsSize = (numGlyphs + 1) * (useLong ? 4 : 2);
  const dataStart = 20 + offsetsSize;
  const out = new Uint8Array(dataStart + dataLength);
  const view = new DataView(out.buffer);
  view.setUint16(0, 1);
  view.setUint16(2, 0);
  view.setUint16(4, axes.length);
  view.setUint16(6, 0); // shared tuple count
  view.setUint32(8, 20); // shared tuples offset (unused)
  view.setUint16(12, numGlyphs);
  view.setUint16(14, useLong ? 1 : 0);
  view.setUint32(16, dataStart);

  for (let gid = 0; gid <= numGlyphs; gid += 1) {
    if (useLong) view.setUint32(20 + gid * 4, offsets[gid]);
    else view.setUint16(20 + gid * 2, offsets[gid] / 2);
  }

  let cursor = dataStart;
  for (let gid = 0; gid < numGlyphs; gid += 1) {
    if (!blocks[gid]) continue;
    out.set(blocks[gid], cursor);
    cursor += blocks[gid].length;
  }
  return out;
}

/**
 * variationsByGlyph: Map<gid, Array<{ peak: Record<string, number>, deltas: Array<[number, number]>, pointCount: number }>>
 */
export function buildVariableFont(font, axes, variationsByGlyph) {
  const names = addNameRecords(font, axes);
  const tables = new Map();
  tables.set("fvar", buildFvar(axes, names.idByText));
  tables.set("avar", buildAvar(axes));
  tables.set("gvar", buildGvar(font.numGlyphs, axes, variationsByGlyph));
  tables.set("name", names.bytes);
  return buildSfnt(font, tables);
}
