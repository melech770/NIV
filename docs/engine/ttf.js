/**
 * Minimal TrueType reader/writer: enough to read glyph outlines and to append
 * fvar/avar/gvar (+ name) tables to a static font.
 */

function tagAt(view, offset) {
  return String.fromCharCode(
    view.getUint8(offset),
    view.getUint8(offset + 1),
    view.getUint8(offset + 2),
    view.getUint8(offset + 3)
  );
}

export function parseFont(buffer) {
  const view = new DataView(buffer);
  const sfntVersion = view.getUint32(0);
  if (sfntVersion !== 0x00010000 && sfntVersion !== 0x74727565 /* 'true' */) {
    throw new Error("unsupported-outline-format");
  }
  const numTables = view.getUint16(4);
  const tables = new Map();
  for (let i = 0; i < numTables; i += 1) {
    const base = 12 + i * 16;
    const tag = tagAt(view, base);
    tables.set(tag, {
      tag,
      checksum: view.getUint32(base + 4),
      offset: view.getUint32(base + 8),
      length: view.getUint32(base + 12),
    });
  }
  if (!tables.has("glyf") || !tables.has("loca") || !tables.has("head") || !tables.has("maxp")) {
    throw new Error("no-glyf");
  }
  if (tables.has("fvar")) {
    throw new Error("already-variable");
  }

  const head = tables.get("head");
  const unitsPerEm = view.getUint16(head.offset + 18);
  const indexToLocFormat = view.getInt16(head.offset + 50);
  const maxp = tables.get("maxp");
  const numGlyphs = view.getUint16(maxp.offset + 4);

  const locaTable = tables.get("loca");
  const loca = new Uint32Array(numGlyphs + 1);
  for (let i = 0; i <= numGlyphs; i += 1) {
    loca[i] =
      indexToLocFormat === 0
        ? view.getUint16(locaTable.offset + i * 2) * 2
        : view.getUint32(locaTable.offset + i * 4);
  }

  const font = {
    buffer,
    view,
    tables,
    sfntVersion,
    numGlyphs,
    unitsPerEm,
    loca,
    cmap: parseCmap(view, tables),
    names: parseNames(view, tables),
    glyphCache: new Map(),
  };
  return font;
}

function parseCmap(view, tables) {
  const table = tables.get("cmap");
  if (!table) return new Map();
  const base = table.offset;
  const count = view.getUint16(base + 2);
  const candidates = [];
  for (let i = 0; i < count; i += 1) {
    const rec = base + 4 + i * 8;
    const platform = view.getUint16(rec);
    const encoding = view.getUint16(rec + 2);
    const offset = base + view.getUint32(rec + 4);
    const format = view.getUint16(offset);
    let score = -1;
    if (format === 12 && platform === 3 && encoding === 10) score = 5;
    else if (format === 12 && platform === 0) score = 4;
    else if (format === 4 && platform === 3 && encoding === 1) score = 3;
    else if (format === 4 && platform === 0) score = 2;
    else if (format === 4) score = 1;
    if (score >= 0) candidates.push({ score, offset, format });
  }
  candidates.sort((a, b) => b.score - a.score);
  const cmap = new Map();
  for (const candidate of candidates) {
    if (candidate.format === 4) readCmapFormat4(view, candidate.offset, cmap);
    else readCmapFormat12(view, candidate.offset, cmap);
    if (cmap.size > 0) break;
  }
  return cmap;
}

function readCmapFormat4(view, offset, cmap) {
  const segCount = view.getUint16(offset + 6) / 2;
  const endBase = offset + 14;
  const startBase = endBase + segCount * 2 + 2;
  const deltaBase = startBase + segCount * 2;
  const rangeBase = deltaBase + segCount * 2;
  for (let i = 0; i < segCount; i += 1) {
    const end = view.getUint16(endBase + i * 2);
    const start = view.getUint16(startBase + i * 2);
    const delta = view.getInt16(deltaBase + i * 2);
    const rangeOffset = view.getUint16(rangeBase + i * 2);
    for (let cp = start; cp <= end && cp !== 0xffff; cp += 1) {
      let gid;
      if (rangeOffset === 0) {
        gid = (cp + delta) & 0xffff;
      } else {
        const address = rangeBase + i * 2 + rangeOffset + (cp - start) * 2;
        if (address + 1 >= view.byteLength) continue;
        gid = view.getUint16(address);
        if (gid !== 0) gid = (gid + delta) & 0xffff;
      }
      if (gid !== 0) cmap.set(cp, gid);
    }
  }
}

function readCmapFormat12(view, offset, cmap) {
  const groups = view.getUint32(offset + 12);
  for (let i = 0; i < groups; i += 1) {
    const base = offset + 16 + i * 12;
    const start = view.getUint32(base);
    const end = view.getUint32(base + 4);
    const startGid = view.getUint32(base + 8);
    for (let cp = start; cp <= end; cp += 1) {
      cmap.set(cp, startGid + (cp - start));
    }
  }
}

function parseNames(view, tables) {
  const table = tables.get("name");
  const names = new Map();
  if (!table) return names;
  const base = table.offset;
  const count = view.getUint16(base + 2);
  const strings = base + view.getUint16(base + 4);
  const records = [];
  for (let i = 0; i < count; i += 1) {
    const rec = base + 6 + i * 12;
    const record = {
      platformID: view.getUint16(rec),
      encodingID: view.getUint16(rec + 2),
      languageID: view.getUint16(rec + 4),
      nameID: view.getUint16(rec + 6),
      length: view.getUint16(rec + 8),
      offset: view.getUint16(rec + 10),
    };
    records.push(record);
    if (!names.has(record.nameID) && (record.platformID === 3 || record.platformID === 0)) {
      names.set(record.nameID, decodeName(view, strings + record.offset, record));
    }
  }
  return { records, strings, names };
}

function decodeName(view, offset, record) {
  const bytes = new Uint8Array(view.buffer, view.byteOffset + offset, record.length);
  if (record.platformID === 0 || record.platformID === 3) {
    let out = "";
    for (let i = 0; i + 1 < bytes.length; i += 2) {
      out += String.fromCharCode((bytes[i] << 8) | bytes[i + 1]);
    }
    return out;
  }
  return Array.from(bytes, (b) => String.fromCharCode(b)).join("");
}

/** Returns the glyph outline, or null for empty/composite glyphs. */
export function readGlyph(font, gid) {
  if (font.glyphCache.has(gid)) return font.glyphCache.get(gid);
  const result = parseGlyph(font, gid);
  font.glyphCache.set(gid, result);
  return result;
}

function parseGlyph(font, gid) {
  const start = font.loca[gid];
  const end = font.loca[gid + 1];
  if (end <= start) return null;
  const glyf = font.tables.get("glyf");
  const view = font.view;
  let p = glyf.offset + start;
  const contours = view.getInt16(p);
  if (contours <= 0) return null; // empty or composite

  p += 10;
  const ends = [];
  for (let i = 0; i < contours; i += 1) {
    ends.push(view.getUint16(p));
    p += 2;
  }
  const instructions = view.getUint16(p);
  p += 2 + instructions;

  const nPoints = ends[contours - 1] + 1;
  const flags = new Uint8Array(nPoints);
  for (let i = 0; i < nPoints; ) {
    const flag = view.getUint8(p);
    p += 1;
    flags[i] = flag;
    i += 1;
    if (flag & 0x08) {
      let repeat = view.getUint8(p);
      p += 1;
      while (repeat > 0 && i < nPoints) {
        flags[i] = flag;
        i += 1;
        repeat -= 1;
      }
    }
  }

  const points = new Array(nPoints);
  let x = 0;
  for (let i = 0; i < nPoints; i += 1) {
    const flag = flags[i];
    if (flag & 0x02) {
      const delta = view.getUint8(p);
      p += 1;
      x += flag & 0x10 ? delta : -delta;
    } else if (!(flag & 0x10)) {
      x += view.getInt16(p);
      p += 2;
    }
    points[i] = { x, y: 0, onCurve: (flag & 0x01) === 1 };
  }
  let y = 0;
  for (let i = 0; i < nPoints; i += 1) {
    const flag = flags[i];
    if (flag & 0x04) {
      const delta = view.getUint8(p);
      p += 1;
      y += flag & 0x20 ? delta : -delta;
    } else if (!(flag & 0x20)) {
      y += view.getInt16(p);
      p += 2;
    }
    points[i].y = y;
  }
  return { points, ends };
}

/** Contour id per point and normalized position inside its contour. */
export function contourInfo(points, ends) {
  const contourIds = new Int32Array(points.length);
  const pos = new Float32Array(points.length);
  let start = 0;
  for (let c = 0; c < ends.length; c += 1) {
    const stop = ends[c] + 1;
    const length = stop - start;
    const denom = Math.max(1, length - 1);
    for (let i = 0; i < length; i += 1) {
      contourIds[start + i] = c;
      pos[start + i] = i / denom;
    }
    start = stop;
  }
  return { contourIds, pos };
}

/* ------------------------------------------------------------------ output */

export function f2dot14(value) {
  return Math.round(value * 16384) & 0xffff;
}

export function addNameRecords(font, entries) {
  const table = font.tables.get("name");
  if (!table) throw new Error("missing-name-table");
  const { records } = font.names;
  const used = new Set(records.map((r) => r.nameID));
  let nextId = 256;
  const idByText = new Map();

  // Copy the original string storage verbatim so existing records keep working.
  const storageStart = font.names.strings - table.offset;
  const originalStorage = new Uint8Array(font.buffer, table.offset + storageStart, table.length - storageStart).slice();

  const appended = [];
  let appendedLength = 0;
  for (const text of entries) {
    if (idByText.has(text)) continue;
    while (used.has(nextId)) nextId += 1;
    used.add(nextId);
    idByText.set(text, nextId);
    const bytes = new Uint8Array(text.length * 2);
    for (let i = 0; i < text.length; i += 1) {
      const code = text.charCodeAt(i);
      bytes[i * 2] = code >> 8;
      bytes[i * 2 + 1] = code & 0xff;
    }
    appended.push({ nameID: nextId, bytes, offset: originalStorage.length + appendedLength });
    appendedLength += bytes.length + (bytes.length % 2);
    nextId += 1;
  }

  const allRecords = records.concat(
    appended.map((item) => ({
      platformID: 3,
      encodingID: 1,
      languageID: 0x0409,
      nameID: item.nameID,
      length: item.bytes.length,
      offset: item.offset,
    }))
  );
  const headerSize = 6 + allRecords.length * 12;
  const out = new Uint8Array(headerSize + originalStorage.length + appendedLength);
  const view = new DataView(out.buffer);
  view.setUint16(0, 0); // format 0
  view.setUint16(2, allRecords.length);
  view.setUint16(4, headerSize);
  allRecords.forEach((record, index) => {
    const base = 6 + index * 12;
    view.setUint16(base, record.platformID);
    view.setUint16(base + 2, record.encodingID);
    view.setUint16(base + 4, record.languageID);
    view.setUint16(base + 6, record.nameID);
    view.setUint16(base + 8, record.length);
    view.setUint16(base + 10, record.offset);
  });
  out.set(originalStorage, headerSize);
  for (const item of appended) {
    out.set(item.bytes, headerSize + item.offset);
  }
  return { bytes: out, idByText };
}

function tableChecksum(bytes) {
  let sum = 0;
  const padded = bytes.length + ((4 - (bytes.length % 4)) % 4);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.length);
  for (let i = 0; i < padded; i += 4) {
    let word = 0;
    for (let j = 0; j < 4; j += 1) {
      const index = i + j;
      word = (word << 8) | (index < bytes.length ? view.getUint8(index) : 0);
    }
    sum = (sum + (word >>> 0)) >>> 0;
  }
  return sum >>> 0;
}

export function buildSfnt(font, newTables) {
  const entries = [];
  for (const [tag, table] of font.tables) {
    if (newTables.has(tag)) continue;
    entries.push({
      tag,
      data: new Uint8Array(font.buffer, table.offset, table.length).slice(),
    });
  }
  for (const [tag, data] of newTables) {
    entries.push({ tag, data });
  }
  entries.sort((a, b) => (a.tag < b.tag ? -1 : a.tag > b.tag ? 1 : 0));

  const headEntry = entries.find((e) => e.tag === "head");
  const headAdjustOffset = 8;
  headEntry.data[headAdjustOffset] = 0;
  headEntry.data[headAdjustOffset + 1] = 0;
  headEntry.data[headAdjustOffset + 2] = 0;
  headEntry.data[headAdjustOffset + 3] = 0;

  const numTables = entries.length;
  const headerSize = 12 + numTables * 16;
  let offset = headerSize;
  for (const entry of entries) {
    entry.offset = offset;
    entry.checksum = tableChecksum(entry.data);
    offset += entry.data.length + ((4 - (entry.data.length % 4)) % 4);
  }
  const total = offset;

  const out = new Uint8Array(total);
  const view = new DataView(out.buffer);
  view.setUint32(0, font.sfntVersion);
  view.setUint16(4, numTables);
  const entrySelector = Math.floor(Math.log2(numTables));
  view.setUint16(6, 16 * 2 ** entrySelector);
  view.setUint16(8, entrySelector);
  view.setUint16(10, numTables * 16 - 16 * 2 ** entrySelector);

  entries.forEach((entry, index) => {
    const base = 12 + index * 16;
    for (let i = 0; i < 4; i += 1) view.setUint8(base + i, entry.tag.charCodeAt(i));
    view.setUint32(base + 4, entry.checksum);
    view.setUint32(base + 8, entry.offset);
    view.setUint32(base + 12, entry.data.length);
    out.set(entry.data, entry.offset);
  });

  const fileChecksum = tableChecksum(out);
  const adjustment = (0xb1b0afba - fileChecksum) >>> 0;
  const headView = new DataView(out.buffer, headEntry.offset, headEntry.data.length);
  headView.setUint32(headAdjustOffset, adjustment);
  return out;
}

