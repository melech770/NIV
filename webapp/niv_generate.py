"""Thin, service-friendly wrapper around the NIV inference pipeline.

The research code in ``code/ttf_to_var.py`` prints one line per glyph and runs the
whole font in a single call.  This module reproduces the same algorithm
(fvar/avar/gvar + model-predicted deltas) with progress reporting, cancellation
and a structured result, so it can back a web service.
"""

from __future__ import annotations

import json
import re
import sys
import threading
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import torch
from fontTools.otlLib.builder import buildStatTable
from fontTools.ttLib import TTFont, newTable
from fontTools.ttLib.tables._f_v_a_r import Axis
from fontTools.ttLib.tables._g_v_a_r import TupleVariation

CODE_DIR = Path(__file__).resolve().parent.parent / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from niv_model import GlyphDeltaRegressor  # noqa: E402

MODEL_AXES: Tuple[str, ...] = ("wght", "wdth", "slnt", "opsz")

AXIS_INFO = {
    "wght": {"label": "משקל", "label_en": "Weight", "hint": "דק ↔ שמן"},
    "wdth": {"label": "רוחב", "label_en": "Width", "hint": "צר ↔ רחב"},
    "slnt": {"label": "הטיה", "label_en": "Slant", "hint": "זקוף ↔ נטוי"},
    "opsz": {"label": "גודל אופטי", "label_en": "Optical size", "hint": "טקסט ↔ כותרות"},
}

# Roughly 10 ms per model forward pass on a 2-core CPU container.
SECONDS_PER_TUPLE = 0.012


class GenerationCancelled(RuntimeError):
    pass


class FontError(ValueError):
    """User-facing problem with the uploaded font or settings."""


@dataclass
class ModelCfg:
    d_model: int
    num_layers: int
    num_heads: int
    ff_mult: int
    dropout: float
    cond_method: str
    pos_enc: str
    point_mlp_hidden: int
    cond_mlp_hidden: int
    max_points: int
    axes: List[str]
    point_contour_info: int
    axisembed_method: str
    prediction_type: str
    split_method: str
    combinatorial_max_order: int
    polar_regression: bool = False
    lineloss: Optional[float] = None
    coordnorm: bool = False


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

_MODEL_CACHE: Dict[str, Tuple[GlyphDeltaRegressor, ModelCfg, int]] = {}
_MODEL_LOCK = threading.Lock()


def _resolve_checkpoint(model_dir: Path) -> Tuple[Path, Path]:
    model_pt = model_dir / "model.pt"
    cfg_path = model_dir / "config.json"
    if not model_pt.exists():
        raise FontError(f"מודל חסר: {model_pt}")
    if not cfg_path.exists():
        raise FontError(f"config.json חסר: {cfg_path}")
    return model_pt, cfg_path


def _infer_from_state_dict(sd: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    point_feat_dim = None
    for k, v in sd.items():
        if k.endswith("point_mlp.0.weight") and getattr(v, "ndim", 0) == 2:
            point_feat_dim = int(v.shape[1])
            break
    if point_feat_dim is None:
        candidates = [
            int(v.shape[1]) for v in sd.values() if getattr(v, "ndim", 0) == 2 and 2 <= int(v.shape[1]) <= 16
        ]
        if not candidates:
            raise FontError("לא ניתן להסיק את מבנה המודל מה-checkpoint.")
        point_feat_dim = min(candidates)

    contour_rows = 0
    for k, v in sd.items():
        if (
            "contour" in k.lower()
            and "embed" in k.lower()
            and k.endswith("weight")
            and getattr(v, "ndim", 0) == 2
        ):
            contour_rows = int(v.shape[0])
            break
    return int(point_feat_dim), contour_rows


def load_model(model_dir: Path) -> Tuple[GlyphDeltaRegressor, ModelCfg, int]:
    """Load (and cache) the NIV checkpoint. Returns model, config, contour count."""
    key = str(Path(model_dir).resolve())
    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached

        model_pt, cfg_path = _resolve_checkpoint(Path(model_dir))
        cfg_raw = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg_raw.pop("loss1", None)
        cfg = ModelCfg(**cfg_raw)

        sd = torch.load(model_pt, map_location="cpu")
        if not isinstance(sd, dict):
            raise FontError("ה-checkpoint אינו בפורמט המצופה.")

        point_feat_dim, contour_rows = _infer_from_state_dict(sd)
        model = GlyphDeltaRegressor(
            num_axes=len(cfg.axes),
            max_points=int(cfg.max_points),
            d_model=int(cfg.d_model),
            num_layers=int(cfg.num_layers),
            num_heads=int(cfg.num_heads),
            ff_mult=int(cfg.ff_mult),
            dropout=float(cfg.dropout),
            cond_method=str(cfg.cond_method),
            axisembed_method=str(cfg.axisembed_method),
            pos_enc=str(cfg.pos_enc),
            point_mlp_hidden=int(cfg.point_mlp_hidden),
            cond_mlp_hidden=int(cfg.cond_mlp_hidden),
            point_contour_info=int(cfg.point_contour_info),
            num_contour_id_plus_three=int(contour_rows),
            point_feat_dim=int(point_feat_dim),
            polar_regression=bool(cfg.polar_regression),
            coordnorm=bool(cfg.coordnorm),
        )
        if any(k.startswith("encoder.") for k in sd) and not any(
            k.startswith("cond_transformer_encoder.") for k in sd
        ):
            sd = {
                (k.replace("encoder.", "cond_transformer_encoder.", 1) if k.startswith("encoder.") else k): v
                for k, v in sd.items()
            }
        model.load_state_dict(sd, strict=True)
        model.eval()
        covered = max(0, int(contour_rows) - 2)
        _MODEL_CACHE[key] = (model, cfg, covered)
        return model, cfg, covered


# ---------------------------------------------------------------------------
# Unicode helpers
# ---------------------------------------------------------------------------

_RANGE_RE = re.compile(r"^\s*([0-9a-fA-FxXuU+]+)\s*[-–]\s*([0-9a-fA-FxXuU+]+)\s*$")


def _parse_codepoint(token: str) -> int:
    token = token.strip().replace("U+", "").replace("u+", "").replace("0x", "").replace("0X", "")
    if not token:
        raise FontError("טווח תווים לא תקין.")
    try:
        if re.fullmatch(r"[0-9]+", token):
            return int(token, 10)
        return int(token, 16)
    except ValueError as exc:
        raise FontError(f"תו לא מזוהה: {token!r}") from exc


def parse_unicode_ranges(spec: str) -> Set[int]:
    """Parse "0x20-0x7E, 0x5D0-0x5EA, 65" into a set of codepoints."""
    values: Set[int] = set()
    for chunk in re.split(r"[,\n;]+", spec or ""):
        chunk = chunk.strip()
        if not chunk:
            continue
        match = _RANGE_RE.match(chunk)
        if match:
            lo, hi = _parse_codepoint(match.group(1)), _parse_codepoint(match.group(2))
            if lo > hi:
                lo, hi = hi, lo
            if hi - lo > 0x20000:
                raise FontError("הטווח גדול מדי.")
            values.update(range(lo, hi + 1))
        else:
            values.add(_parse_codepoint(chunk))
    if not values:
        raise FontError("לא נבחרו תווים.")
    return values


def _font_cmap(tt: TTFont) -> Dict[int, str]:
    best = tt.getBestCmap() or {}
    return {int(cp): name for cp, name in best.items()}


def _glyph_stats(tt: TTFont, cps: Optional[Sequence[int]] = None) -> Dict[str, int]:
    glyf = tt["glyf"]
    cmap = _font_cmap(tt)
    names = {cmap[cp] for cp in (cps if cps is not None else cmap.keys()) if cp in cmap}
    simple = 0
    composite = 0
    for name in names:
        glyph = glyf.glyphs.get(name)
        if glyph is None:
            continue
        try:
            glyph.expand(glyf)
        except Exception:
            continue
        if glyph.isComposite():
            composite += 1
        elif getattr(glyph, "numberOfContours", 0):
            simple += 1
    return {"mapped": len(names), "simple": simple, "composite": composite}


# ---------------------------------------------------------------------------
# Font inspection
# ---------------------------------------------------------------------------

def font_info(path: Path) -> Dict[str, object]:
    try:
        tt = TTFont(str(path), lazy=True)
    except Exception as exc:  # fontTools raises a variety of errors
        raise FontError("הקובץ אינו פונט TTF/OTF תקין.") from exc

    try:
        if "glyf" not in tt:
            raise FontError(
                "הפונט אינו מבוסס outlines מסוג TrueType (glyf). פונטים בפורמט CFF/OTF אינם נתמכים."
            )
        if "fvar" in tt:
            raise FontError("הפונט הזה כבר פונט משתנה (יש בו צירי fvar), אין צורך להמיר אותו.")

        name = _family_name(tt)
        cmap = _font_cmap(tt)
        stats = _glyph_stats(tt)
        presets = {
            "latin": _count_in(tt, parse_unicode_ranges("0x20-0x7E")),
            "hebrew": _count_in(tt, parse_unicode_ranges("0x5D0-0x5F4")),
            "latin_hebrew": _count_in(tt, parse_unicode_ranges("0x20-0x7E,0x5D0-0x5F4")),
            "all": stats,
        }
        return {
            "name": name,
            "units_per_em": int(tt["head"].unitsPerEm),
            "glyphs": len(tt.getGlyphOrder()),
            "mapped": stats["mapped"],
            "simple": stats["simple"],
            "composite": stats["composite"],
            "has_gvar": "gvar" in tt,
            "presets": presets,
            "axes": AXIS_INFO,
        }
    finally:
        tt.close()


def _count_in(tt: TTFont, cps: Set[int]) -> Dict[str, int]:
    cmap = _font_cmap(tt)
    return _glyph_stats(tt, [cp for cp in cps if cp in cmap])


def _family_name(tt: TTFont) -> str:
    try:
        for rec in tt["name"].names:
            if rec.nameID in (16, 1):
                try:
                    value = rec.toUnicode().strip()
                except Exception:
                    continue
                if value:
                    return value
    except Exception:
        pass
    return "ללא שם"


# ---------------------------------------------------------------------------
# Table construction
# ---------------------------------------------------------------------------

def _add_fvar(tt: TTFont, axis_tags: Sequence[str]) -> None:
    if "fvar" in tt:
        return
    fvar = newTable("fvar")
    fvar.axes = []
    fvar.instances = []
    name_table = tt["name"]

    def add_name(text: str) -> int:
        for rec in name_table.names:
            try:
                if rec.toUnicode() == text:
                    return int(rec.nameID)
            except Exception:
                continue
        used = {int(r.nameID) for r in name_table.names}
        name_id = 256
        while name_id in used:
            name_id += 1
        name_table.setName(text, name_id, 3, 1, 0x409)
        name_table.setName(text, name_id, 1, 0, 0)
        return name_id

    for tag in axis_tags:
        axis = Axis()
        axis.axisTag = tag
        axis.minValue = -1.0
        axis.defaultValue = 0.0
        axis.maxValue = 1.0
        axis.flags = 0
        axis.axisNameID = add_name(tag)
        fvar.axes.append(axis)
    tt["fvar"] = fvar


def _add_avar_identity(tt: TTFont, axis_tags: Sequence[str]) -> None:
    if not axis_tags or "avar" in tt:
        return
    avar = newTable("avar")
    avar.version = 0x00010000
    avar.segments = {tag: {-1.0: -1.0, 0.0: 0.0, 1.0: 1.0} for tag in axis_tags}
    tt["avar"] = avar


def _add_gvar_empty(tt: TTFont) -> None:
    if "gvar" in tt:
        return
    gvar = newTable("gvar")
    gvar.version = 1
    gvar.variations = {}
    tt["gvar"] = gvar


def _add_stat(tt: TTFont, axis_tags: Sequence[str]) -> None:
    if "STAT" in tt:
        return
    axes = [{"tag": tag, "name": tag, "ordering": i} for i, tag in enumerate(axis_tags)]
    try:
        buildStatTable(tt, axes)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------

def _axis_weight(min_val: float, peak_val: float, max_val: float, value: float) -> float:
    if value <= min_val or value >= max_val:
        return 1.0 if abs(value - peak_val) < 1e-9 else 0.0
    if abs(value - peak_val) < 1e-9:
        return 1.0
    if value < peak_val:
        return 0.0 if abs(peak_val - min_val) < 1e-9 else (value - min_val) / (peak_val - min_val)
    return 0.0 if abs(max_val - peak_val) < 1e-9 else (max_val - value) / (max_val - peak_val)


def _tuple_weight(axes: Dict[str, Tuple[float, float, float]], values: Dict[str, float]) -> float:
    weight = 1.0
    for tag, (lo, peak, hi) in axes.items():
        weight *= _axis_weight(lo, peak, hi, values.get(tag, 0.0))
        if weight == 0.0:
            break
    return weight


def _contour_ids(end_pts: Sequence[int]) -> List[int]:
    ids: List[int] = []
    start = 0
    for index, end in enumerate(end_pts):
        ids.extend([index] * max(0, end - start + 1))
        start = end + 1
    return ids


def _pos_in_contour(contour_ids: Sequence[int]) -> List[float]:
    out = [0.0] * len(contour_ids)
    i = 0
    while i < len(contour_ids):
        j = i + 1
        while j < len(contour_ids) and contour_ids[j] == contour_ids[i]:
            j += 1
        length = j - i
        denom = float(max(1, length - 1))
        for k in range(length):
            out[i + k] = k / denom
        i = j
    return out


@torch.no_grad()
def _predict(
    model: GlyphDeltaRegressor,
    cfg: ModelCfg,
    *,
    points_xy: Sequence[Tuple[int, int]],
    oncurve: Sequence[int],
    contour_ids: List[int],
    axis_ids: Sequence[int],
    axis_values: Sequence[float],
    num_contours: int,
    units_per_em: int,
) -> List[Tuple[int, int]]:
    device = next(model.parameters()).device
    n_base = len(points_xy)
    n = n_base + 4  # four phantom points

    xy = torch.tensor(list(points_xy), dtype=torch.float32) * (2.0 / float(units_per_em)) - 1.0
    oc = torch.tensor(list(oncurve), dtype=torch.float32).view(n_base, 1)
    base = torch.cat([xy, oc], dim=-1)

    phantom = torch.cat(
        [torch.zeros((4, 2), dtype=torch.float32), torch.full((4, 1), -1.0, dtype=torch.float32)], dim=-1
    )
    pts = torch.cat([base, phantom], dim=0)

    cids = torch.tensor(contour_ids, dtype=torch.long)
    tok = torch.empty((n,), dtype=torch.long)
    if num_contours > 0 and n_base > 0:
        tok[:n_base] = torch.clamp(cids, min=0, max=num_contours - 1)
    else:
        tok[:n_base] = 0
    tok[n_base:] = num_contours

    if int(cfg.point_contour_info) in (2, 3):
        extra = torch.zeros((n, 1), dtype=torch.float32)
        if n_base > 0:
            if int(cfg.point_contour_info) == 2:
                extra[0, 0] = 1.0
                if n_base > 1:
                    extra[1:n_base, 0] = (cids[1:] != cids[:-1]).float()
            else:
                extra[:n_base, 0] = torch.tensor(_pos_in_contour(contour_ids), dtype=torch.float32)
        pts = torch.cat([pts, extra], dim=-1)

    pred = model(
        pts.unsqueeze(0).to(device),
        tok.unsqueeze(0).to(device),
        torch.tensor([axis_ids], dtype=torch.long, device=device),
        torch.tensor([axis_values], dtype=torch.float32, device=device),
        torch.zeros((1, n), dtype=torch.bool, device=device),
        torch.ones((1, len(axis_ids)), dtype=torch.bool, device=device),
        combinatorial_max_order=cfg.combinatorial_max_order,
    )[0].to("cpu")

    pred = pred * (float(units_per_em) / 2.0)
    return [
        (int(round(float(pred[j, 0].item()))), int(round(float(pred[j, 1].item())))) for j in range(n)
    ]


def _axis_permutations(axis_tags: Sequence[str], delta_level: int) -> List[Dict[str, object]]:
    perms = []
    for values in product((-1.0, 0.0, 1.0), repeat=len(axis_tags)):
        nonzeros = [i for i, v in enumerate(values) if v != 0.0]
        if not nonzeros or len(nonzeros) > delta_level:
            continue
        perms.append(
            {
                "values": [float(v) for v in values],
                "nonzero_indices": nonzeros,
            }
        )
    perms.sort(key=lambda p: len(p["nonzero_indices"]))
    return perms


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def estimate(tt: TTFont, cps: Set[int], axes: Sequence[str], delta_level: int) -> Dict[str, object]:
    stats = _glyph_stats(tt, [cp for cp in cps if cp in _font_cmap(tt)])
    tuples = stats["simple"] * len(_axis_permutations(axes, delta_level))
    return {
        "glyphs": stats["simple"],
        "composite": stats["composite"],
        "tuples": tuples,
        "seconds": round(1.5 + tuples * SECONDS_PER_TUPLE, 1),
    }


def generate(
    src: Path,
    out: Path,
    model_dir: Path,
    *,
    axes: Sequence[str],
    cps: Set[int],
    delta_level: int = 1,
    progress: Optional[Callable[[float, str, Dict[str, object]], None]] = None,
    cancelled: Optional[threading.Event] = None,
) -> Dict[str, object]:
    """Generate a variable font from a static TTF. Mirrors code/ttf_to_var.py."""
    axes = [a for a in axes if a in MODEL_AXES]
    if not axes:
        raise FontError("יש לבחור לפחות ציר אחד.")
    if not 1 <= delta_level <= len(axes):
        raise FontError("רמת דלתא חייבת להיות בין 1 למספר הצירים.")

    started = time.perf_counter()
    model, cfg, num_contours = load_model(model_dir)
    axis_to_id = {tag: i for i, tag in enumerate(cfg.axes)}
    missing = [a for a in axes if a not in axis_to_id]
    if missing:
        raise FontError(f"המודל אינו תומך בצירים: {', '.join(missing)}")

    tt = TTFont(str(src))
    try:
        if "glyf" not in tt:
            raise FontError("הפונט אינו מבוסס glyf.")
        perms = _axis_permutations(axes, delta_level)

        _add_fvar(tt, axes)
        _add_avar_identity(tt, axes)
        _add_gvar_empty(tt)
        _add_stat(tt, axes)

        glyf = tt["glyf"]
        gvar = tt["gvar"]
        cmap = _font_cmap(tt)

        items = sorted((cp, name) for cp, name in cmap.items() if cp in cps)
        total = len(items) * len(perms)
        done = 0
        tuples_added = 0
        skipped_composite = 0
        processed_glyphs = 0
        upm = int(tt["head"].unitsPerEm)

        def report(message: str, force: bool = False) -> None:
            if progress is None:
                return
            fraction = (done / total) if total else 1.0
            progress(fraction, message, {"done": done, "total": total})

        for cp, name in items:
            if cancelled is not None and cancelled.is_set():
                raise GenerationCancelled("הפעולה בוטלה.")
            glyph = glyf.glyphs.get(name)
            if glyph is None:
                done += len(perms)
                continue
            try:
                glyph.expand(glyf)
            except Exception:
                done += len(perms)
                continue
            if glyph.isComposite():
                skipped_composite += 1
                done += len(perms)
                continue

            coords, end_pts, flags = glyph.getCoordinates(glyf)
            end_pts = list(end_pts) if end_pts is not None else []
            if not end_pts:
                done += len(perms)
                continue

            n_base = int(end_pts[-1]) + 1
            points_xy = [(int(coords[i][0]), int(coords[i][1])) for i in range(n_base)]
            oncurve = [1 if (int(flags[i]) & 0x01) else 0 for i in range(n_base)]
            ids = _contour_ids(end_pts)
            if len(ids) != n_base:
                done += len(perms)
                continue

            existing = gvar.variations.setdefault(name, [])
            processed_glyphs += 1

            for perm in perms:
                if cancelled is not None and cancelled.is_set():
                    raise GenerationCancelled("הפעולה בוטלה.")

                axes_indices = perm["nonzero_indices"]
                values = perm["values"]
                deltas_subtract = None

                if existing:
                    axis_values = {axes[i]: values[i] for i in axes_indices}
                    accumulated = [(0.0, 0.0)] * (n_base + 4)
                    for variation in existing:
                        weight = _tuple_weight(
                            {tag: (lo, peak, hi) for tag, (lo, peak, hi) in variation.axes.items()},
                            axis_values,
                        )
                        if weight >= 0:
                            for idx, (dx, dy) in enumerate(variation.coordinates):
                                acc_x, acc_y = accumulated[idx]
                                accumulated[idx] = (acc_x + weight * dx, acc_y + weight * dy)
                    deltas_subtract = accumulated

                deltas = _predict(
                    model,
                    cfg,
                    points_xy=points_xy,
                    oncurve=oncurve,
                    contour_ids=ids,
                    axis_ids=[axis_to_id[axes[i]] for i in axes_indices],
                    axis_values=[float(values[i]) for i in axes_indices],
                    num_contours=num_contours,
                    units_per_em=upm,
                )

                if deltas_subtract is not None:
                    deltas = [
                        (
                            int(round(dx - deltas_subtract[idx][0])),
                            int(round(dy - deltas_subtract[idx][1])),
                        )
                        for idx, (dx, dy) in enumerate(deltas)
                    ]

                tuple_axes: Dict[str, Tuple[float, float, float]] = {}
                for axis_index in axes_indices:
                    peak = float(values[axis_index])
                    tuple_axes[axes[axis_index]] = (0.0, peak, peak) if peak > 0 else (peak, peak, 0.0)

                existing.append(TupleVariation(tuple_axes, deltas))
                tuples_added += 1
                done += 1
                if progress is not None and (done % 5 == 0 or done == total):
                    report(f"מחשב דלתות עבור {name}…")

        report("כותב את הפונט המשתנה…", force=True)
        tt.save(str(out))
    finally:
        tt.close()

    elapsed = time.perf_counter() - started
    return {
        "glyphs": processed_glyphs,
        "tuples": tuples_added,
        "skipped_composite": skipped_composite,
        "codepoints": len(items),
        "seconds": round(elapsed, 1),
        "axes": axes,
        "delta_level": delta_level,
        "out": str(out),
    }
