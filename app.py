"""
ISTIAX Downloader API
---------------------
A small Flask API that downloads MP4 video or extracts MP3 audio with yt-dlp.
Use it only for media you own or are authorized to download.

Endpoints
  GET /api/health
  GET /api/download?url=<URL>&format=mp4|mp3      (needs API key)

Auth: send the key in the "X-API-Key" header (preferred) or ?api_key=KEY.
Secrets are read from environment variables only. Nothing is hard-coded.
"""

import hmac
import ipaddress
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

from flask import Flask, jsonify, request, send_file
from werkzeug.exceptions import HTTPException

APP_NAME = "ISTIAX Downloader API"
APP_VERSION = "1.0.0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("istiax")


# --------------------------------------------------------------------------
# Configuration (all from environment variables, with safe defaults)
# --------------------------------------------------------------------------
def _env_int(name, default, minimum, maximum):
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


MAX_FILESIZE_MB = _env_int("MAX_FILESIZE_MB", 50, 1, 500)
MAX_PROCESS_SECONDS = _env_int("MAX_PROCESS_SECONDS", 90, 10, 600)
MAX_HEIGHT = _env_int("MAX_HEIGHT", 720, 144, 2160)
MAX_CONCURRENT_JOBS = _env_int("MAX_CONCURRENT_JOBS", 1, 1, 4)
MAX_BYTES = MAX_FILESIZE_MB * 1024 * 1024
STALE_JOB_SECONDS = 30 * 60

# The "generic" extractor lets yt-dlp fetch arbitrary URLs. It is OFF by
# default because it widens the attack surface (SSRF). Turn it on only if you
# need direct links to .mp4/.mp3 files.
ALLOW_GENERIC = os.environ.get("ALLOW_GENERIC_EXTRACTOR", "false").strip().lower() in (
    "1",
    "true",
    "yes",
)

JOB_ROOT = Path(tempfile.gettempdir()) / "istiax_jobs"
JOB_ROOT.mkdir(parents=True, exist_ok=True)

MIME_TYPES = {"mp4": "video/mp4", "mp3": "audio/mpeg"}

_job_slots = threading.BoundedSemaphore(MAX_CONCURRENT_JOBS)

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------
class ApiError(Exception):
    def __init__(self, status, code, message):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


@app.errorhandler(ApiError)
def handle_api_error(err):
    resp = jsonify({"error": err.code, "message": err.message})
    resp.status_code = err.status
    if err.status == 429:
        resp.headers["Retry-After"] = "15"
    return resp


@app.errorhandler(Exception)
def handle_unexpected(err):
    if isinstance(err, HTTPException):
        resp = jsonify({"error": err.name.lower().replace(" ", "_"), "message": err.description})
        resp.status_code = err.code or 500
        return resp
    log.exception("Unhandled error")
    resp = jsonify({"error": "server_error", "message": "Internal server error."})
    resp.status_code = 500
    return resp


@app.after_request
def add_security_headers(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Cache-Control"] = "no-store"
    return resp


# --------------------------------------------------------------------------
# FFmpeg discovery
# --------------------------------------------------------------------------
def _setup_ffmpeg():
    """Return a directory containing an executable named 'ffmpeg', or None.

    1. Use a system ffmpeg if there is one.
    2. Otherwise use the static binary bundled by the 'imageio-ffmpeg' package
       (this is what makes Render's plain Python runtime work).
    """
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return str(Path(system_ffmpeg).parent)
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        bin_dir = Path(tempfile.gettempdir()) / "istiax_ffmpeg"
        bin_dir.mkdir(parents=True, exist_ok=True)
        link = bin_dir / "ffmpeg"
        if link.is_symlink() or link.exists():
            link.unlink()
        try:
            link.symlink_to(exe)
        except OSError:
            shutil.copy2(exe, link)
            os.chmod(link, 0o755)
        return str(bin_dir)
    except Exception as exc:  # noqa: BLE001
        log.warning("FFmpeg not available: %s", exc)
        return None


FFMPEG_DIR = _setup_ffmpeg()


# --------------------------------------------------------------------------
# Authentication
# --------------------------------------------------------------------------
def _configured_api_key():
    return os.environ.get("API_KEY", "")


def require_api_key():
    expected = _configured_api_key()
    if len(expected) < 16:
        # Fail closed: never run open if the key is missing or weak.
        raise ApiError(
            503,
            "server_not_configured",
            "API_KEY is not set (or is shorter than 16 characters) on the server.",
        )
    provided = request.headers.get("X-API-Key") or request.args.get("api_key") or ""
    if not hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8")):
        raise ApiError(401, "unauthorized", "Missing or invalid API key.")


# --------------------------------------------------------------------------
# URL validation (SSRF protection)
# --------------------------------------------------------------------------
def _allowed_domains():
    raw = os.environ.get("ALLOWED_DOMAINS", "")
    return [d.strip().lower().lstrip(".") for d in raw.split(",") if d.strip()]


def _ip_is_public(ip):
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


def validate_url(raw):
    if not raw or not raw.strip():
        raise ApiError(400, "missing_url", "The 'url' parameter is required.")
    raw = raw.strip()
    if len(raw) > 2048:
        raise ApiError(400, "invalid_url", "URL is too long.")

    try:
        parsed = urlparse(raw)
        port = parsed.port
    except ValueError:
        raise ApiError(400, "invalid_url", "URL could not be parsed.")

    if parsed.scheme not in ("http", "https"):
        raise ApiError(400, "invalid_url", "Only http:// and https:// URLs are allowed.")
    if parsed.username or parsed.password:
        raise ApiError(400, "invalid_url", "URLs with embedded credentials are not allowed.")
    host = (parsed.hostname or "").lower()
    if not host:
        raise ApiError(400, "invalid_url", "URL has no hostname.")
    if port not in (None, 80, 443):
        raise ApiError(400, "invalid_url", "Only ports 80 and 443 are allowed.")
    if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
        raise ApiError(400, "blocked_destination", "Local destinations are not allowed.")

    allowed = _allowed_domains()
    if allowed and not any(host == d or host.endswith("." + d) for d in allowed):
        raise ApiError(403, "domain_not_allowed", "This domain is not in the server's allow-list.")

    try:
        infos = socket.getaddrinfo(
            host, port or (443 if parsed.scheme == "https" else 80), proto=socket.IPPROTO_TCP
        )
    except socket.gaierror:
        raise ApiError(400, "unresolvable_host", "Could not resolve the hostname.")

    if not infos:
        raise ApiError(400, "unresolvable_host", "Could not resolve the hostname.")
    for info in infos:
        address = info[4][0].split("%")[0]
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            raise ApiError(400, "blocked_destination", "Unrecognised address.")
        if not _ip_is_public(ip):
            raise ApiError(
                400,
                "blocked_destination",
                "Private, local, or reserved IP destinations are not allowed.",
            )
    return raw, host


# --------------------------------------------------------------------------
# yt-dlp runner (separate process so it can be killed on timeout / size cap)
# --------------------------------------------------------------------------
def build_command(url, fmt, media_dir):
    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--ignore-config",
        "--no-playlist",
        "--no-cache-dir",
        "--no-mtime",
        "--restrict-filenames",
        "--socket-timeout",
        "15",
        "--retries",
        "2",
        "--fragment-retries",
        "2",
        "--max-filesize",
        f"{MAX_FILESIZE_MB}M",
        "--match-filters",
        "!is_live",
        "-o",
        str(media_dir / "%(title).80s.%(ext)s"),
    ]
    if FFMPEG_DIR:
        cmd += ["--ffmpeg-location", FFMPEG_DIR]
    if not ALLOW_GENERIC:
        cmd += ["--use-extractors", "default,-generic"]

    if fmt == "mp4":
        h = MAX_HEIGHT
        selector = (
            f"bv*[height<={h}][ext=mp4]+ba[ext=m4a]/b[height<={h}][ext=mp4]/"
            f"bv*[height<={h}]+ba/b[height<={h}]"
        )
        cmd += ["-f", selector, "--merge-output-format", "mp4"]
    else:
        cmd += ["-f", "bestaudio/best", "-x", "--audio-format", "mp3", "--audio-quality", "192K"]

    cmd += ["--", url]
    return cmd


def _child_env(home):
    # Deliberately minimal: the child process never sees API_KEY or other secrets.
    return {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": str(home),
        "TMPDIR": str(home),
        "LANG": "C.UTF-8",
        "PYTHONUNBUFFERED": "1",
    }


def _largest_file_size(directory):
    biggest = 0
    try:
        for p in directory.iterdir():
            try:
                if p.is_file():
                    biggest = max(biggest, p.stat().st_size)
            except OSError:
                continue
    except OSError:
        pass
    return biggest


def _kill_group(proc):
    try:
        if hasattr(os, "killpg"):
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        proc.wait(timeout=5)
    except Exception:  # noqa: BLE001
        pass


def run_yt_dlp(cmd, media_dir, log_path, workdir):
    """Run yt-dlp; return 'ok', 'failed', 'timeout' or 'too_large'."""
    deadline = time.monotonic() + MAX_PROCESS_SECONDS
    with open(log_path, "wb") as log_file:
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            cwd=str(media_dir),
            env=_child_env(workdir),
            start_new_session=True,
        )
        try:
            while True:
                try:
                    code = proc.wait(timeout=0.5)
                    return "ok" if code == 0 else "failed"
                except subprocess.TimeoutExpired:
                    pass
                if time.monotonic() > deadline:
                    _kill_group(proc)
                    return "timeout"
                if _largest_file_size(media_dir) > MAX_BYTES:
                    _kill_group(proc)
                    return "too_large"
        finally:
            if proc.poll() is None:
                _kill_group(proc)


def _read_log_tail(log_path, limit=8192):
    try:
        with open(log_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - limit))
            return f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _last_error_line(log_text, workdir):
    for line in reversed(log_text.splitlines()):
        if "ERROR:" in line:
            return line.split("ERROR:", 1)[1].strip().replace(str(workdir), "")[:300]
    return "Unknown error."


def find_output_file(media_dir, fmt):
    """Return the finished file, making sure it is a real file inside media_dir."""
    root = media_dir.resolve()
    candidates = []
    for p in media_dir.iterdir():
        if p.is_symlink() or not p.is_file():
            continue
        if p.suffix.lower() != f".{fmt}":
            continue
        if p.resolve().parent != root:
            continue
        candidates.append(p)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_size)


# --------------------------------------------------------------------------
# Temp-file housekeeping
# --------------------------------------------------------------------------
def _finish_job(workdir):
    try:
        if workdir:
            shutil.rmtree(workdir, ignore_errors=True)
    finally:
        try:
            _job_slots.release()
        except ValueError:
            pass


def sweep_stale_jobs():
    """Remove leftover job folders (e.g. after a crash)."""
    now = time.time()
    try:
        for d in JOB_ROOT.iterdir():
            try:
                if d.is_dir() and now - d.stat().st_mtime > STALE_JOB_SECONDS:
                    shutil.rmtree(d, ignore_errors=True)
            except OSError:
                continue
    except OSError:
        pass


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/")
def index():
    return jsonify(
        {
            "service": APP_NAME,
            "version": APP_VERSION,
            "endpoints": ["/api/health", "/api/download?url=URL&format=mp4|mp3"],
        }
    )


@app.get("/api/health")
def health():
    try:
        from yt_dlp.version import __version__ as ytdlp_version
    except Exception:  # noqa: BLE001
        ytdlp_version = None
    ffmpeg_ok = FFMPEG_DIR is not None
    key_ok = len(_configured_api_key()) >= 16
    return jsonify(
        {
            "status": "ok" if (ffmpeg_ok and key_ok) else "degraded",
            "service": APP_NAME,
            "version": APP_VERSION,
            "yt_dlp_version": ytdlp_version,
            "ffmpeg_available": ffmpeg_ok,
            "api_key_configured": key_ok,
            "limits": {
                "max_filesize_mb": MAX_FILESIZE_MB,
                "max_process_seconds": MAX_PROCESS_SECONDS,
                "max_video_height": MAX_HEIGHT,
            },
        }
    )


@app.get("/api/download")
def download():
    require_api_key()

    fmt = (request.args.get("format") or "mp4").strip().lower()
    if fmt not in MIME_TYPES:
        raise ApiError(400, "invalid_format", "format must be 'mp4' or 'mp3'.")

    url, host = validate_url(request.args.get("url"))

    if fmt == "mp3" and FFMPEG_DIR is None:
        raise ApiError(503, "ffmpeg_missing", "FFmpeg is not available on this server.")

    if not _job_slots.acquire(blocking=False):
        raise ApiError(429, "busy", "The server is busy with another download. Try again shortly.")

    workdir = None
    handed_off = False
    try:
        sweep_stale_jobs()
        workdir = Path(tempfile.mkdtemp(prefix="job_", dir=JOB_ROOT))
        media_dir = workdir / "media"
        media_dir.mkdir()
        log_path = workdir / "yt-dlp.log"

        log.info("job start format=%s host=%s", fmt, host)
        outcome = run_yt_dlp(build_command(url, fmt, media_dir), media_dir, log_path, workdir)

        if outcome == "timeout":
            raise ApiError(
                504, "timeout", f"Processing exceeded the {MAX_PROCESS_SECONDS}s limit."
            )
        if outcome == "too_large":
            raise ApiError(413, "too_large", f"File exceeds the {MAX_FILESIZE_MB} MB limit.")

        log_text = _read_log_tail(log_path)
        if outcome == "failed":
            raise ApiError(
                502,
                "download_failed",
                "yt-dlp could not download this media: " + _last_error_line(log_text, workdir),
            )

        out_file = find_output_file(media_dir, fmt)
        if out_file is None:
            if "max-filesize" in log_text:
                raise ApiError(413, "too_large", f"File exceeds the {MAX_FILESIZE_MB} MB limit.")
            if "does not pass filter" in log_text:
                raise ApiError(422, "rejected", "This media was rejected (for example a live stream).")
            raise ApiError(502, "no_output", "No output file was produced.")

        size = out_file.stat().st_size
        if size == 0:
            raise ApiError(502, "empty_output", "The output file was empty.")
        if size > MAX_BYTES:
            raise ApiError(413, "too_large", f"File exceeds the {MAX_FILESIZE_MB} MB limit.")

        response = send_file(
            out_file,
            mimetype=MIME_TYPES[fmt],
            as_attachment=True,
            download_name=out_file.name,
            max_age=0,
        )
        # Delete the temp folder only after the response has been fully sent.
        response.call_on_close(lambda: _finish_job(workdir))
        handed_off = True
        log.info("job ok format=%s host=%s bytes=%d", fmt, host, size)
        return response
    finally:
        if not handed_off:
            _finish_job(workdir)


if __name__ == "__main__":
    # Local development / fallback. On Render, gunicorn is used (see render.yaml).
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
