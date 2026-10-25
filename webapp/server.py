"""FastAPI service for the NIV variable-font generator."""

from __future__ import annotations

import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Set

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import niv_generate as niv
from fontTools.ttLib import TTFont

ROOT = Path(__file__).resolve().parent.parent
WEBAPP = Path(__file__).resolve().parent
DATA = WEBAPP / "data"
UPLOADS = DATA / "uploads"
OUTPUTS = DATA / "outputs"
STATIC = WEBAPP / "static"
MODEL_DIR = ROOT / "model"

for folder in (UPLOADS, OUTPUTS):
    folder.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
JOB_TTL_SECONDS = 2 * 60 * 60

app = FastAPI(title="NIV · Font to Variable Font", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="niv")
_lock = threading.Lock()
_jobs: Dict[str, "Job"] = {}
_sources: Dict[str, Path] = {}
_source_names: Dict[str, str] = {}


@dataclass
class Job:
    id: str
    token: str
    params: dict
    state: str = "queued"
    progress: float = 0.0
    message: str = "בתור…"
    result: Optional[dict] = None
    error: Optional[str] = None
    created: float = field(default_factory=time.time)
    started: Optional[float] = None
    finished: Optional[float] = None
    cancel: threading.Event = field(default_factory=threading.Event)

    def as_dict(self) -> dict:
        payload = {
            "id": self.id,
            "state": self.state,
            "progress": round(self.progress, 4),
            "message": self.message,
            "error": self.error,
            "params": self.params,
        }
        if self.result:
            payload["result"] = self.result
            payload["fontUrl"] = f"/api/fonts/{self.id}.ttf"
            payload["downloadUrl"] = f"/api/download/{self.id}"
        if self.started and not self.finished:
            elapsed = time.time() - self.started
            if 0 < self.progress < 1:
                payload["elapsed"] = round(elapsed, 1)
                payload["eta"] = round(max(0.0, elapsed * (1 - self.progress) / self.progress), 1)
            else:
                payload["elapsed"] = round(elapsed, 1)
        return payload


class GenerateRequest(BaseModel):
    token: str
    axes: list[str] = Field(default_factory=lambda: ["wght"])
    delta_level: int = 1
    preset: str = "latin"
    custom: str = ""


class EstimateRequest(BaseModel):
    token: str
    axes: list[str] = Field(default_factory=lambda: ["wght"])
    delta_level: int = 1
    preset: str = "latin"
    custom: str = ""


def _safe_name(name: str) -> str:
    name = Path(name or "font").name
    name = re.sub(r"[^A-Za-z0-9._\-\u0590-\u05FF ]+", "_", name).strip()
    return name or "font"


def _source_path(token: str) -> Path:
    path = _sources.get(token)
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="הפונט לא נמצא. העלה אותו שוב.")
    return path


def _resolve_codepoints(tt: TTFont, preset: str, custom: str) -> Set[int]:
    if preset == "custom":
        return niv.parse_unicode_ranges(custom)
    if preset == "all":
        return set(niv._font_cmap(tt).keys())
    table = {
        "latin": "0x20-0x7E",
        "hebrew": "0x5D0-0x5F4",
        "latin_hebrew": "0x20-0x7E,0x5D0-0x5F4",
    }
    if preset not in table:
        raise HTTPException(status_code=400, detail="בחירת תווים לא מוכרת.")
    return niv.parse_unicode_ranges(table[preset])


def _cleanup() -> None:
    cutoff = time.time() - JOB_TTL_SECONDS
    with _lock:
        stale = [job_id for job_id, job in _jobs.items() if job.created < cutoff and job.state != "running"]
    for job_id in stale:
        with _lock:
            _jobs.pop(job_id, None)
        (OUTPUTS / f"{job_id}.ttf").unlink(missing_ok=True)
        (OUTPUTS / f"{job_id}.name").unlink(missing_ok=True)

    upload_cutoff = time.time() - 6 * JOB_TTL_SECONDS
    for upload in UPLOADS.iterdir():
        try:
            if upload.is_file() and upload.stat().st_mtime < upload_cutoff:
                upload.unlink()
        except OSError:
            continue


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "model": (MODEL_DIR / "model.pt").exists(), "device": "cpu"}


@app.post("/api/inspect")
def inspect(file: UploadFile = File(...)) -> dict:
    raw = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="הקובץ גדול מדי (מקסימום 25MB).")
    if len(raw) < 12:
        raise HTTPException(status_code=400, detail="הקובץ ריק או פגום.")
    if raw[:4] not in (b"\x00\x01\x00\x00", b"true", b"OTTO", b"ttcf"):
        raise HTTPException(status_code=400, detail="זה לא נראה כמו קובץ פונט (TTF/OTF).")

    token = uuid.uuid4().hex
    original = _safe_name(file.filename or "font.ttf")
    if not original.lower().endswith((".ttf", ".otf")):
        original += ".ttf"
    path = UPLOADS / f"{token}-{original}"
    path.write_bytes(raw)

    try:
        info = niv.font_info(path)
    except niv.FontError as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"לא ניתן לקרוא את הפונט: {exc}") from exc

    with _lock:
        _sources[token] = path
        _source_names[token] = original
    info["token"] = token
    info["filename"] = original
    return info


def _estimate_for(req: EstimateRequest) -> dict:
    path = _source_path(req.token)
    axes = [a for a in req.axes if a in niv.MODEL_AXES]
    if not axes:
        return {"glyphs": 0, "tuples": 0, "seconds": 0, "composite": 0}
    tt = TTFont(str(path))
    try:
        cps = _resolve_codepoints(tt, req.preset, req.custom)
        return niv.estimate(tt, cps, axes, max(1, min(4, int(req.delta_level))))
    except niv.FontError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        tt.close()


@app.post("/api/estimate")
def estimate(req: EstimateRequest) -> dict:
    return _estimate_for(req)


def _run_job(job: Job) -> None:
    job.state = "running"
    job.started = time.time()
    job.message = "טוען את המודל…"
    try:
        source = _source_path(job.token)
        params = job.params
        tt = TTFont(str(source))
        try:
            cps = _resolve_codepoints(tt, params["preset"], params["custom"])
        finally:
            tt.close()

        original = _source_names.get(job.token) or source.name
        stem = _safe_name(original).rsplit(".", 1)[0]
        out = OUTPUTS / f"{job.id}.ttf"
        out.with_name(f"{job.id}.name").write_text(f"{stem}-variable.ttf", encoding="utf-8")

        def on_progress(fraction: float, message: str, _stats: dict) -> None:
            job.progress = max(job.progress, min(fraction, 0.99))
            job.message = message

        result = niv.generate(
            source,
            out,
            MODEL_DIR,
            axes=[a for a in params["axes"] if a in niv.MODEL_AXES],
            cps=cps,
            delta_level=int(params["delta_level"]),
            progress=on_progress,
            cancelled=job.cancel,
        )
        result["chars"] = "".join(chr(cp) for cp in sorted(cps) if 0x20 <= cp <= 0x2FFF)[:400]
        result["bytes"] = out.stat().st_size
        job.progress = 1.0
        job.state = "done"
        job.message = "הפונט המשתנה מוכן"
        job.result = result
        job.finished = time.time()
    except niv.GenerationCancelled:
        job.state = "cancelled"
        job.message = "הפעולה בוטלה"
        job.finished = time.time()
        (OUTPUTS / f"{job.id}.ttf").unlink(missing_ok=True)
    except niv.FontError as exc:
        job.state = "error"
        job.error = str(exc)
        job.message = "שגיאה"
        job.finished = time.time()
    except Exception as exc:  # pragma: no cover - unexpected model/font failure
        job.state = "error"
        job.error = f"{type(exc).__name__}: {exc}"
        job.message = "שגיאה"
        job.finished = time.time()


@app.post("/api/generate")
def generate(req: GenerateRequest) -> dict:
    _cleanup()
    source = _source_path(req.token)
    axes = [a for a in req.axes if a in niv.MODEL_AXES]
    if not axes:
        raise HTTPException(status_code=400, detail="יש לבחור לפחות ציר אחד (wght / wdth / slnt / opsz).")
    delta_level = max(1, min(len(axes), int(req.delta_level)))

    with _lock:
        active = [j for j in _jobs.values() if j.state in ("queued", "running")]
        if len(active) >= 2:
            raise HTTPException(status_code=429, detail="יש כבר שתי משימות בתור. נסה שוב בעוד רגע.")
        job = Job(
            id=uuid.uuid4().hex,
            token=req.token,
            params={
                "axes": axes,
                "delta_level": delta_level,
                "preset": req.preset,
                "custom": req.custom,
                "font": _source_names.get(req.token, source.name),
            },
        )
        _jobs[job.id] = job

    _executor.submit(_run_job, job)
    return job.as_dict()


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="המשימה לא נמצאה.")
    return job.as_dict()


@app.post("/api/jobs/{job_id}/cancel")
def job_cancel(job_id: str) -> dict:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="המשימה לא נמצאה.")
    job.cancel.set()
    return {"ok": True}


@app.get("/api/source/{token}.ttf")
def source_font(token: str) -> FileResponse:
    path = _source_path(token)
    return FileResponse(path, media_type="font/ttf", headers={"Cache-Control": "no-store"})


@app.get("/api/fonts/{job_id}.ttf")
def job_font(job_id: str) -> FileResponse:
    path = OUTPUTS / f"{job_id}.ttf"
    if not path.exists():
        raise HTTPException(status_code=404, detail="הפונט עדיין לא נוצר.")
    return FileResponse(
        path,
        media_type="font/ttf",
        headers={"Cache-Control": "no-store", "Content-Disposition": f'inline; filename="{job_id}.ttf"'},
    )


@app.get("/api/download/{job_id}")
def job_download(job_id: str) -> FileResponse:
    path = OUTPUTS / f"{job_id}.ttf"
    if not path.exists():
        raise HTTPException(status_code=404, detail="הפונט עדיין לא נוצר.")
    name_file = OUTPUTS / f"{job_id}.name"
    name = name_file.read_text(encoding="utf-8").strip() if name_file.exists() else "font-variable.ttf"
    return FileResponse(path, media_type="font/ttf", filename=name)


@app.exception_handler(niv.FontError)
def font_error_handler(_request, exc: niv.FontError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "webapp.server:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "3000")),
        log_level="info",
    )
