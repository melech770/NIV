"""Export the NIV checkpoint to ONNX for in-browser inference.

The research model conditions its transformer through a Python loop over axis
combinations (``AxisIdAndValueEmbedder``), which cannot be traced.  The export
wrapper recomputes that conditioning with plain tensor ops, so the browser can
pass pre-computed combination indices and weights for any axis tuple.

Usage:
    python tools/export_onnx.py --out docs/models/niv.onnx
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from fontTools.ttLib import TTFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from webapp import niv_generate as G  # noqa: E402
from tools.onnx_wrapper import NivOnnxModule  # noqa: E402


def combo_table(num_axes: int) -> Dict[Tuple[Tuple[int, int], ...], int]:
    """Replicates AxisIdAndValueEmbedder.combo_to_idx."""
    table: Dict[Tuple[Tuple[int, int], ...], int] = {}
    index = 0
    for depth in range(1, num_axes + 1):
        for axis_combo in combinations(range(num_axes), depth):
            for signs in range(2**depth):
                sign_list = [(1 if (signs >> i) & 1 else -1) for i in range(depth)]
                key = tuple((axis_combo[i], sign_list[i]) for i in range(depth))
                table[key] = index
                index += 1
    return table


def conditioning_terms(
    axes: Sequence[str],
    axis_to_id: Dict[str, int],
    values: Sequence[float],
    table: Dict[Tuple[Tuple[int, int], ...], int],
    max_order: int,
) -> Tuple[List[int], List[float]]:
    """Same terms as AxisIdAndValueEmbedder._get_combinatorial_terms_for_item."""
    active = sorted((axis_to_id[axes[i]], float(values[i])) for i in range(len(axes)) if values[i] != 0.0)
    indices: List[int] = []
    weights: List[float] = []
    for depth in range(1, min(len(active), max_order) + 1):
        for subset in combinations(range(len(active)), depth):
            key = []
            weight = 1.0
            for pos in subset:
                axis_id, axis_value = active[pos]
                key.append((axis_id, 1 if axis_value >= 0 else -1))
                weight *= abs(axis_value)
            indices.append(table[tuple(key)])
            weights.append(weight)
    return indices, weights


def glyph_points(tt: TTFont, codepoint: int) -> np.ndarray | None:
    glyf = tt["glyf"]
    name = (tt.getBestCmap() or {}).get(codepoint)
    if name is None:
        return None
    glyph = glyf.glyphs[name]
    glyph.expand(glyf)
    if glyph.isComposite():
        return None
    coords, end_pts, flags = glyph.getCoordinates(glyf)
    end_pts = list(end_pts) if end_pts is not None else []
    if not end_pts:
        return None
    upm = int(tt["head"].unitsPerEm)
    count = int(end_pts[-1]) + 1
    pos = np.asarray(G._pos_in_contour(G._contour_ids(end_pts)), dtype=np.float32)
    xy = np.asarray([(int(coords[i][0]), int(coords[i][1])) for i in range(count)], dtype=np.float32)
    xy = xy * (2.0 / upm) - 1.0
    oncurve = np.asarray([1.0 if (int(flags[i]) & 1) else 0.0 for i in range(count)], dtype=np.float32)
    out = np.zeros((count + 4, 4), dtype=np.float32)
    out[:count, 0:2] = xy
    out[:count, 2] = oncurve
    out[:count, 3] = pos
    out[count:, 2] = -1.0
    return out


def batch(arrays: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = max(array.shape[0] for array in arrays)
    points = np.zeros((len(arrays), n, 4), dtype=np.float32)
    mask = np.zeros((len(arrays), n), dtype=bool)
    for row, array in enumerate(arrays):
        points[row, : array.shape[0]] = array
        mask[row, array.shape[0] :] = True
    positions = np.broadcast_to(np.arange(n, dtype=np.float32), (len(arrays), n)).copy()
    return points, mask, positions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "model"))
    ap.add_argument("--font", default=str(ROOT / "samples/Heebo-Regular-static.ttf"))
    ap.add_argument("--out", default=str(ROOT / "docs/models/niv.onnx"))
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    model, cfg, _ = G.load_model(Path(args.model))
    module = NivOnnxModule(model).eval()
    table = combo_table(len(cfg.axes))
    axis_to_id = {tag: i for i, tag in enumerate(cfg.axes)}

    tt = TTFont(args.font)
    arrays = [a for a in (glyph_points(tt, cp) for cp in (0x41, 0x67, 0x5D0, 0x5EA, 0x31)) if a is not None]
    if not arrays:
        raise SystemExit("no simple glyphs in parity font")
    points, mask, positions = batch(arrays)
    print(f"parity batch: {points.shape[0]} glyphs, sizes={[a.shape[0] for a in arrays]}")

    combos = [(["wght"], [1.0]), (["wght"], [-1.0]), (["wght", "wdth"], [1.0, -1.0])]

    # reference values from the untouched research forward
    references = []
    with torch.no_grad():
        for axes, values in combos:
            references.append(
                model(
                    torch.from_numpy(points),
                    torch.zeros((points.shape[0], points.shape[1]), dtype=torch.long),
                    torch.tensor([[axis_to_id[a] for a in axes]]),
                    torch.tensor([values], dtype=torch.float32),
                    torch.from_numpy(mask),
                    torch.ones((1, len(axes)), dtype=torch.bool),
                    combinatorial_max_order=cfg.combinatorial_max_order,
                ).numpy()
            )

    inputs = (
        torch.from_numpy(points),
        torch.from_numpy(mask),
        torch.from_numpy(positions),
        torch.tensor([0], dtype=torch.long),
        torch.tensor([1.0], dtype=torch.float32),
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print("exporting to", out_path)
    torch.onnx.export(
        module,
        inputs,
        str(out_path),
        input_names=["points", "pad_mask", "positions", "combo_idx", "combo_w"],
        output_names=["deltas"],
        dynamic_axes={
            "points": {0: "batch", 1: "points"},
            "pad_mask": {0: "batch", 1: "points"},
            "positions": {0: "batch", 1: "points"},
            "combo_idx": {0: "terms"},
            "combo_w": {0: "terms"},
            "deltas": {0: "batch", 1: "points"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )

    import onnxruntime as ort

    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    print("inputs:", [(i.name, i.shape) for i in session.get_inputs()])
    for (axes, values), reference in zip(combos, references):
        indices, weights = conditioning_terms(axes, axis_to_id, values, table, cfg.combinatorial_max_order)
        output = session.run(
            None,
            {
                "points": points,
                "pad_mask": mask,
                "positions": positions,
                "combo_idx": np.asarray(indices, dtype=np.int64),
                "combo_w": np.asarray(weights, dtype=np.float32),
            },
        )[0]
        diff = float(np.abs(output - reference).max())
        print(f"{axes} {values}: max |onnx - torch| = {diff:.6f} (normalized units)")

    print(f"onnx size: {out_path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
