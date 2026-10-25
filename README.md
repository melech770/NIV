<div align="center">

# NIV: Neural Axis Variations for Variable Font Generation

Nadav Benedek, Ariel Shamir, Ohad Fried

<br>

<a href="https://arxiv.org/abs/2606.05261"><strong>Paper</strong></a>
&nbsp;|&nbsp;
<a href="https://ndvbd.github.io/NIV/"><strong>Project Page</strong></a>

</div>

---

This repository is the official implementation of: NIV: Neural Axis Variations for Variable Font Generation.

## Structure

- `environment.yml`: Conda environment specification.
- `code/`: Python source files for training and generation.
- `model/`: Trained model checkpoint folder (used by `ttf_to_var.py`)
- `webapp/`: FastAPI service + browser UI for turning a static font into a variable font.
- `dataset_sample/`: Sample of the prepared dataset (split files, dataset license, and a small set of XML files).
  To generate the full dataset, run the two commands in **Quick Start**:
  1. `data_get_variable_google_fonts.py`
  2. `data_prepare_training_data.py`

## Environment Setup

Set up the conda environment. This may take a few minutes.

```bash
conda env create -f environment.yml
conda activate niv_env
```


## Quick Start


### Download variable fonts

Use this script to clone the Google Fonts repository into google-fonts subdirectory, and extract only variable-font TTF files into google-fonts-variable. This takes a few minutes.

```bash
python code/data_get_variable_google_fonts.py
```

### Prepare training data

Use this step to convert and preprocess a folder of `.ttf` fonts into XML training files in the dataset folder. This should take around 15 minutes.

Example:

```bash
python code/data_prepare_training_data.py \
  --fonts-dir google-fonts-variable \
  --out-dir dataset \
  --skip-composites \
  --normalize-upm \
  --only-axes wght,wdth,slnt,opsz
```

Alternatively, you can download the prepared dataset from Hugging Face:

```bash
pip install -U huggingface_hub
hf download ndvb/NIV \
  --repo-type dataset \
  --local-dir dataset
```

### Training

Train the model using the dataset directory and the predefined train/test split which is specified in dataset_sample for consistent results. The default argument for best performance is to load dataset into RAM (around 50GB of RAM).
You can skip this step and use a trained model.

```bash
python code/niv_model_train.py \
  --data-dir dataset \
  --fixed-split-dir dataset_sample \
  --seed 42 \
  --output-dir model_output \
  --train-loss mse \
  --eval-loss mse \
  --split-method font \
  --eval-steps 20000 \
  --only-axes wght,wdth,slnt,opsz \
  --epochs 80
```
### Evaluation

Evaluate on font-split:

```bash
python code/niv_model_train.py \
  --data-dir dataset \
  --fixed-split-dir dataset_sample \
  --load-checkpoint model \
  --seed 42 \
  --train-loss mse \
  --eval-loss mse \
  --split-method font \
  --only-axes wght,wdth,slnt,opsz \
  --only-eval
```


### Generate a variable font

```bash
python code/ttf_to_var.py /path/to/font.ttf --model /path/to/model/best --axes wght,wdth,slnt,opsz --unicode-range 0x20-0x7E --delta-level 4
```

Output is saved next to the source font as `*_var.ttf`.
You can use a model you trained, or take a (font-split) trained model from `model/`.

## Web tool: upload a font, get a variable font

A static, server-less version of the tool lives in `docs/` and runs entirely in the browser —
no Python, no uploads, no GPU. The checkpoint is exported to ONNX (`docs/models/niv.onnx`) and
runs on WASM through `onnxruntime-web`; the font is parsed and re-serialized in JavaScript.

```bash
npm install                       # only for the browser runtime tests
python -m http.server 3000 -d docs # or any static server / GitHub Pages
# open http://localhost:3000
```

### Publishing on GitHub Pages

Everything needed is committed under `docs/` (page, engine, ONNX model, ONNX Runtime WASM,
`.nojekyll`), so Pages needs no build step and no CI:

1. Repository **Settings → Pages**.
2. **Source**: *Deploy from a branch*.
3. **Branch**: `main`, **folder**: `/docs` → **Save**.

After a minute the tool is live at <https://melech770.github.io/NIV/>. All asset paths are
relative, so it also works under the `/NIV/` sub-path. The first visit downloads ~38 MB
(ONNX model + WASM runtime); the browser caches them afterwards. The page has to be served
over HTTP(S) — `file://` will not work.

Regenerating the ONNX model after changing the checkpoint:

```bash
pip install -r requirements-web.txt
python tools/export_onnx.py                  # writes docs/models/niv.onnx
node tools/test_engine.mjs samples/*.ttf     # JS pipeline vs. fontTools
node tools/test_web_runtime.mjs samples/*.ttf  # same, but through the worker + WASM
```

Limits of the browser build: the model is float32 (~26 MB, cached after the first load), and
very large character sets take longer than on a native build. Everything else — axes, delta
level, coverage, the delta math — is identical to the Python pipeline.

The `webapp/` folder is the optional server version: the same pipeline wrapped in a small
FastAPI service with a Hebrew RTL UI. Drop in a static `.ttf`, choose axes and glyph coverage,
and get a ready-to-download variable font with live sliders for every generated axis.

```bash
pip install -r requirements-web.txt
uvicorn webapp.server:app --host 0.0.0.0 --port 3000
# open http://localhost:3000
```

The model checkpoint in `model/` is loaded once and reused for every request; generation runs
in a single background worker with progress, ETA and cancel support.

- **Axes**: `wght`, `wdth`, `slnt`, `opsz` (the four axes the checkpoint was trained on).
- **Delta level**: 1 = single-axis variations, 2-3 add combined tuples (slower, richer).
- **Coverage**: Basic Latin, Hebrew, both, a custom Unicode range (`0x5D0-0x5EA`) or every mapped glyph.
- **Output**: `fvar` axes normalized to `-1..1` with default `0`, identity `avar`, a minimal `STAT`
  table, and model-predicted `gvar` deltas for every simple glyph in the selection.

Limitations inherited from the pipeline: composite glyphs (accents, some Hebrew presentation
forms) keep their static outlines, CFF/OTF outlines are not supported, and fonts that already
contain `fvar` are rejected. Typical runtimes on 2 CPU cores are a few seconds for Basic Latin
with one axis.

Container image:

```bash
docker build -t niv-web . && docker run --rm -p 8000:8000 niv-web
```

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{benedek2026niv,
  title={NIV: Neural Axis Variations for Variable Font Generation},
  author={Benedek, Nadav and Shamir, Ariel and Fried, Ohad},
  journal={arXiv preprint arXiv:2606.05261},
  year={2026}
}
```
