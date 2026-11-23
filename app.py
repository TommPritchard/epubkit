"""
FastAPI application for epubkit — EPUB optimizer for e-ink readers.
Handles: file upload, metadata preview, SSE progress streaming, file download.
Secured for private Reading Ledger integration.
"""

import os
import hmac
import json
import uuid
import shutil
import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request

from epub_processor import process_epub, extract_epub_metadata, ProcessingOptions, ProcessingReport
from image_processor import DEVICE_PROFILES
from metadata_handler import FILENAME_FORMATS, format_filename


def _parse_allowed_origins() -> list[str]:
    raw = os.environ.get("ALLOWED_ORIGINS", "").strip()
    if raw:
        return [o.strip() for o in raw.split(",") if o.strip()]
    return [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8000",
    ]


# Secret API key from environment variable (fail-closed if unset)
CONVERTER_API_KEY = os.environ.get("CONVERTER_API_KEY", "")
ALLOWED_ORIGINS = _parse_allowed_origins()
ALLOWED_ORIGIN_REGEX = os.environ.get(
    "ALLOWED_ORIGIN_REGEX",
    r"^https://([a-zA-Z0-9-]+\.)?vercel\.app$",
)


def _is_authorized(request: Request) -> bool:
    """Verify request API key using constant-time comparison (fail-closed)."""
    expected = (CONVERTER_API_KEY or "").strip()
    if not expected:
        return False
    # TODO(MP-04): drop query_params.get("api_key") once Reading_Ledger MP-04 ships header-only / SSE token auth
    token = request.headers.get("X-API-Key") or request.query_params.get("api_key")
    if not token:
        return False
    return hmac.compare_digest(token.encode("utf-8"), expected.encode("utf-8"))


# Temp directory for uploads and outputs
TEMP_DIR = Path("./tmp")
UPLOAD_DIR = TEMP_DIR / "uploads"
OUTPUT_DIR = TEMP_DIR / "outputs"
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
UPLOAD_CHUNK_SIZE = 1024 * 1024  # 1MB streaming chunks
TASK_TTL_SECONDS = 3600

# Thread pool for processing (capped at 2 for Render's 512 MB free tier)
executor = ThreadPoolExecutor(max_workers=2)

# In-memory task tracking
tasks: dict = {}


def evict_expired_tasks(cutoff: float) -> int:
    """Remove expired task entries older than cutoff timestamp."""
    expired_ids = [
        tid for tid, info in list(tasks.items())
        if info.get("created_at", 0) < cutoff
    ]
    for tid in expired_ids:
        tasks.pop(tid, None)
    return len(expired_ids)


async def cleanup_old_files():
    """Remove temp files and task entries older than 1 hour."""
    while True:
        await asyncio.sleep(300)  # Check every 5 minutes
        cutoff = time.time() - TASK_TTL_SECONDS
        evict_expired_tasks(cutoff)
        for d in [UPLOAD_DIR, OUTPUT_DIR]:
            if d.exists():
                for item in d.iterdir():
                    if item.is_dir() and item.stat().st_mtime < cutoff:
                        shutil.rmtree(item, ignore_errors=True)
        batch_dir = OUTPUT_DIR / "batch"
        if batch_dir.exists():
            for item in batch_dir.iterdir():
                if item.is_file() and item.stat().st_mtime < cutoff:
                    item.unlink(missing_ok=True)


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cleanup_task = asyncio.create_task(cleanup_old_files())
    try:
        yield
    finally:
        cleanup_task.cancel()


app = FastAPI(title="epubkit", lifespan=lifespan)

# 1. Enable CORS for Reading Ledger web app + localhost dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=ALLOWED_ORIGIN_REGEX,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["X-API-Key", "Content-Type"],
)


# 2. Security Middleware: fail-closed API Key verification on all non-health requests
@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    # Always allow CORS preflight requests
    if request.method == "OPTIONS":
        return await call_next(request)

    # Health check is open for waking up the container
    if request.url.path == "/health":
        return await call_next(request)

    if not _is_authorized(request):
        # Return 404 to mask the service from public visitors
        return HTMLResponse(
            "<!DOCTYPE html><html><head><title>404 Not Found</title></head><body><h1>404 Not Found</h1></body></html>",
            status_code=404
        )

    return await call_next(request)


# Health check endpoint for wake-up ping
@app.get("/health")
async def health():
    return {"status": "ok", "service": "reading-ledger-converter"}


# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    if not _is_authorized(request):
        return HTMLResponse("<!DOCTYPE html><html><head><title>404 Not Found</title></head><body><h1>404 Not Found</h1></body></html>", status_code=404)
    return templates.TemplateResponse("index.html", {"request": request})


def _sanitize_upload_filename(raw_filename: str | None) -> str:
    """Strip directory components (POSIX and Windows) from uploaded filenames."""
    if not raw_filename:
        return ""
    name = Path(raw_filename.replace("\\", "/")).name.strip()
    if name in ("", ".", ".."):
        return ""
    return name


@app.post("/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    """Upload one or more EPUB files. Returns task IDs and metadata for each."""
    results = []

    for file in files:
        safe_filename = _sanitize_upload_filename(file.filename)
        if not safe_filename or not safe_filename.lower().endswith('.epub'):
            results.append({
                "filename": safe_filename or file.filename,
                "error": "Not an EPUB file",
                "task_id": None,
            })
            continue

        # Check for .kepub.epub
        if safe_filename.lower().endswith('.kepub.epub'):
            results.append({
                "filename": safe_filename,
                "error": "Kobo EPUB files (.kepub.epub) are not supported",
                "task_id": None,
            })
            continue

        task_id = str(uuid.uuid4())
        task_dir = UPLOAD_DIR / task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        # Save uploaded file via bounded streaming to avoid loading full file into RAM
        file_path = task_dir / safe_filename
        if file_path.resolve().parent != task_dir.resolve():
            shutil.rmtree(task_dir, ignore_errors=True)
            results.append({
                "filename": safe_filename,
                "error": "Invalid filename",
                "task_id": None,
            })
            continue

        total_size = 0
        too_large = False
        with open(file_path, 'wb') as f:
            while True:
                chunk = await file.read(UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                total_size += len(chunk)
                if total_size > MAX_FILE_SIZE:
                    too_large = True
                    break
                f.write(chunk)

        if too_large:
            shutil.rmtree(task_dir, ignore_errors=True)
            results.append({
                "filename": safe_filename,
                "error": f"File too large (max {MAX_FILE_SIZE // (1024*1024)}MB)",
                "task_id": None,
            })
            continue

        # Extract metadata for preview
        try:
            metadata = extract_epub_metadata(str(file_path))
        except Exception as e:
            metadata = {"title": "", "author": "", "error": str(e)}

        tasks[task_id] = {
            "status": "uploaded",
            "filename": safe_filename,
            "file_path": str(file_path),
            "file_size": total_size,
            "created_at": time.time(),
        }

        results.append({
            "filename": safe_filename,
            "task_id": task_id,
            "file_size": total_size,
            "metadata": metadata,
            "error": metadata.get("error"),
        })

    return {"files": results}


@app.get("/process/{task_id}")
async def process_sse(
    task_id: str,
    device: str = "x4",
    grayscale: bool = True,
    contrast: bool = True,
    quality: int = 70,
    remove_fonts: bool = True,
    remove_css: bool = True,
    light_novel: bool = False,
    generate_cover: bool = True,
    clean_metadata: bool = True,
    text_cleanup: bool = True,
    edit_title: str = "",
    edit_author: str = "",
    filename_format: str = "author-title",
    filename_template: str = "",
    api_key: str = ""
):
    """SSE endpoint that streams processing progress."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = tasks[task_id]
    if task["status"] == "processing":
        raise HTTPException(status_code=409, detail="Already processing")

    if device not in DEVICE_PROFILES:
        allowed = ", ".join(f"'{d}'" for d in DEVICE_PROFILES)
        raise HTTPException(status_code=400, detail=f"Unknown device (expected {allowed})")
    if filename_format not in FILENAME_FORMATS:
        raise HTTPException(status_code=400, detail="Unknown filename format")
    if filename_format == "custom" and not filename_template.strip():
        raise HTTPException(status_code=400, detail="Custom filename template cannot be empty")
    if len(filename_template) > 200:
        raise HTTPException(status_code=400, detail="Filename template is too long")
    if filename_format == "custom":
        try:
            format_filename(
                "title",
                "author",
                filename_format="custom",
                template=filename_template,
                original_filename="original.epub",
                year="2026",
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    input_path = task["file_path"]
    out_dir = OUTPUT_DIR / task_id
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(out_dir / "output.epub")

    options = ProcessingOptions(
        device=device,
        grayscale=grayscale,
        contrast_boost=contrast,
        quality=quality,
        remove_fonts=remove_fonts,
        remove_unused_css=remove_css,
        light_novel_mode=light_novel,
        generate_missing_cover=generate_cover,
        clean_metadata=clean_metadata,
        text_cleanup=text_cleanup,
        filename_format=filename_format,
        filename_template=filename_template,
        original_filename=task["filename"],
    )

    if edit_title or edit_author:
        options.metadata_edits = {}
        if edit_title:
            options.metadata_edits['title'] = edit_title
        if edit_author:
            options.metadata_edits['author'] = edit_author

    task["status"] = "processing"

    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def progress_callback(percent: int, message: str):
        loop.call_soon_threadsafe(queue.put_nowait, {
            "percent": percent,
            "message": message,
        })

    async def generate():
        future = loop.run_in_executor(
            executor,
            process_epub,
            input_path,
            output_path,
            options,
            progress_callback,
        )

        done = False
        while not done:
            try:
                update = await asyncio.wait_for(queue.get(), timeout=0.5)
                yield f"data: {json.dumps(update)}\n\n"
                if update.get("percent", 0) >= 100:
                    done = True
            except asyncio.TimeoutError:
                yield f": keepalive\n\n"

        report: ProcessingReport = await future

        final = {
            "percent": 100,
            "message": "Complete" if report.success else f"Error: {report.error}",
            "status": "done" if report.success else "error",
            "report": {
                "success": report.success,
                "error": report.error,
                "original_size": report.original_size,
                "optimized_size": report.optimized_size,
                "output_filename": report.output_filename,
                "summary": report.summary(),
                "images_converted": report.images_converted,
                "images_total": report.images_total,
                "fonts_removed": report.fonts_removed,
                "css_rules_removed": report.css_rules_removed,
                "svg_covers_fixed": report.svg_covers_fixed,
                "toc_status": report.toc_status,
                "metadata_items_stripped": report.metadata_items_stripped,
                "cover_generated": report.cover_generated,
                "attrs_stripped": report.attrs_stripped,
                "text_fixes_total": report.text_fixes_total,
                "text_cleanup_summary": report.text_cleanup_summary,
            }
        }

        task["status"] = "done" if report.success else "error"
        task["report"] = final["report"]
        task["output_path"] = output_path
        task["output_filename"] = report.output_filename

        yield f"data: {json.dumps(final)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.get("/download/{task_id}")
async def download_file(task_id: str, api_key: str = ""):
    """Download the processed EPUB file."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = tasks[task_id]
    if task["status"] != "done":
        raise HTTPException(status_code=400, detail="Processing not complete")

    output_path = task.get("output_path")
    if not output_path or not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Output file not found")

    filename = task.get("output_filename", "optimized.epub")

    return FileResponse(
        output_path,
        media_type="application/epub+zip",
        filename=filename,
    )


@app.get("/download-all")
async def download_all(task_ids: str, api_key: str = ""):
    """Download multiple processed EPUBs as a ZIP. task_ids is comma-separated."""
    import zipfile
    ids = [t.strip() for t in task_ids.split(',') if t.strip()]

    if not ids:
        raise HTTPException(status_code=400, detail="No task IDs provided")

    zip_dir = OUTPUT_DIR / "batch"
    zip_dir.mkdir(parents=True, exist_ok=True)
    zip_path = str(zip_dir / f"epubkit_optimized_{uuid.uuid4().hex[:8]}.zip")

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for task_id in ids:
            if task_id in tasks and tasks[task_id]["status"] == "done":
                task = tasks[task_id]
                output_path = task.get("output_path", "")
                filename = task.get("output_filename", "optimized.epub")
                if os.path.exists(output_path):
                    zf.write(output_path, filename)

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename="epubkit_optimized_epubs.zip",
    )


if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get('PORT', 8000))
    uvicorn.run(app, host='0.0.0.0', port=port)
