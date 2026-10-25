/** Font summary used by the UI before generation. */
import { readGlyph } from "./ttf.js";

export const PRESET_RANGES = {
  latin: [[0x20, 0x7e]],
  hebrew: [[0x5d0, 0x5f4]],
  latin_hebrew: [
    [0x20, 0x7e],
    [0x5d0, 0x5f4],
  ],
};

export function presetCodepoints(preset) {
  const out = [];
  for (const [lo, hi] of PRESET_RANGES[preset] || []) {
    for (let cp = lo; cp <= hi; cp += 1) out.push(cp);
  }
  return out;
}

export const AXIS_INFO = {
  wght: { label: "משקל", hint: "דק ↔ שמן" },
  wdth: { label: "רוחב", hint: "צר ↔ רחב" },
  slnt: { label: "הטיה", hint: "זקוף ↔ נטוי" },
  opsz: { label: "גודל אופטי", hint: "טקסט ↔ כותרות" },
};

export function familyName(font) {
  const names = font.names.names;
  return names.get(16) || names.get(1) || "ללא שם";
}

function countCoverage(font, codepoints) {
  let simple = 0;
  let composite = 0;
  let mapped = 0;
  const seen = new Set();
  for (const cp of codepoints) {
    const gid = font.cmap.get(cp);
    if (gid === undefined || seen.has(gid)) continue;
    seen.add(gid);
    mapped += 1;
    const outline = readGlyph(font, gid);
    if (outline) simple += 1;
    else composite += 1;
  }
  return { mapped, simple, composite };
}

export function inspectFont(font) {
  const allCodepoints = [...font.cmap.keys()];
  const all = countCoverage(font, allCodepoints);
  const presets = {
    latin: countCoverage(font, presetCodepoints("latin")),
    hebrew: countCoverage(font, presetCodepoints("hebrew")),
    latin_hebrew: countCoverage(font, presetCodepoints("latin_hebrew")),
    all,
  };
  return {
    name: familyName(font),
    unitsPerEm: font.unitsPerEm,
    glyphs: font.numGlyphs,
    mapped: all.mapped,
    simple: all.simple,
    composite: all.composite,
    presets,
    axes: AXIS_INFO,
  };
}
