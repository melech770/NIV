/** Unicode range spec parser: "0x20-0x7E", "U+5D0-5EA", "65, 66-68". */
export function parseRangeSpec(spec) {
  const values = new Set();
  const decode = (token) => {
    const cleaned = token.trim().replace(/^u\+/i, "").replace(/^0x/i, "");
    if (!cleaned) throw new Error("טווח תווים לא תקין");
    const value = /^[0-9]+$/.test(cleaned) ? parseInt(cleaned, 10) : parseInt(cleaned, 16);
    if (!Number.isFinite(value) || value < 0) throw new Error(`תו לא מזוהה: ${token}`);
    return value;
  };
  for (const chunk of String(spec || "").split(/[,;\n]+/)) {
    const piece = chunk.trim();
    if (!piece) continue;
    const match = piece.match(/^([0-9a-fA-FuUxX+]+)\s*[-–]\s*([0-9a-fA-FuUxX+]+)$/);
    if (match) {
      let lo = decode(match[1]);
      let hi = decode(match[2]);
      if (lo > hi) [lo, hi] = [hi, lo];
      if (hi - lo > 0x20000) throw new Error("הטווח גדול מדי");
      for (let cp = lo; cp <= hi; cp += 1) values.add(cp);
    } else {
      values.add(decode(piece));
    }
  }
  if (!values.size) throw new Error("לא נבחרו תווים");
  return values;
}
