"""Local HTTP interface to the existing transcription, score and piano renderer."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import json
import math
import mimetypes
from pathlib import Path
import secrets
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlsplit
import wave
import webbrowser

import torch
from matplotlib.figure import Figure

from .inference import load_model, transcribe_audio
from .score import draw_staff
from .synthesis import piano_sound_name, render_result_wav
from .viewer import draw_prediction, read_prediction, waveform_envelope

PROJECT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).with_name("web_static")
PREFERRED = ("piano-v8.1.pt", "piano-v9.pt", "piano-v8.pt", "piano-v7.pt", "piano-v6.pt", "piano-v5.pt")


class WebState:
    def __init__(self, project=PROJECT, site_origins=()):
        self.project = Path(project).resolve()
        self.site_origins = frozenset(site_origins)
        self.temporary = tempfile.TemporaryDirectory(prefix=".piano-web-", dir=self.project)
        self.directory = Path(self.temporary.name)
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.RLock()
        self.plot_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="piano-web")
        self.models, self.model_names, self.audio, self.results, self.jobs = {}, {}, {}, {}, {}
        self.refresh_models()

    def close(self):
        with self.lock:
            for job in self.jobs.values():
                job["cancel"].set()
        self.executor.shutdown(wait=True)
        self.temporary.cleanup()

    def refresh_models(self):
        with self.lock:
            for path in sorted((self.project / "checkpoints").glob("*.pt")):
                if path not in self.models.values():
                    self.models[secrets.token_hex(12)] = path
            available = [{"id": key, "name": self.model_names.get(key, path.name)} for key, path in self.models.items() if path.is_file()]
        available.sort(key=lambda row: (PREFERRED.index(row["name"]) if row["name"] in PREFERRED else len(PREFERRED), row["name"]))
        return available

    def lookup(self, collection, key):
        with self.lock:
            if key not in collection:
                raise KeyError("Item not found. Upload or select the file again.")
            return collection[key]

    def model_details(self, key):
        path = self.lookup(self.models, key)
        try:
            model = load_model(path, torch.device("cpu"))
        except Exception as error:
            raise ValueError("Cannot open this checkpoint. Choose a compatible piano model .pt file.") from error
        return {"id": key, "name": self.model_names.get(key, path.name), "architecture": model.architecture,
                "version": getattr(model, "model_version", None),
                "parameters": sum(p.numel() for p in model.parameters()),
                "thresholds": model.decoding_thresholds.summary(),
                "pitch_dependent": model.decoding_thresholds.pitch_dependent}

    def register_audio(self, path, name):
        with wave.open(str(path), "rb") as source:
            if source.getsampwidth() != 2 or source.getcomptype() != "NONE" or source.getnchannels() not in (1, 2):
                raise ValueError("Choose a mono or stereo uncompressed 16-bit PCM WAV.")
            duration = source.getnframes() / source.getframerate()
            if duration <= 0:
                raise ValueError("The audio recording is empty.")
        key = secrets.token_hex(12)
        item = {"id": key, "path": path, "name": name, "duration": duration}
        with self.lock:
            self.audio[key] = item
        return {"id": key, "name": name, "duration": duration, "url": f"/api/audio/{key}/file"}

    def add_result(self, result, audio_id=None):
        key = secrets.token_hex(12)
        with self.lock:
            self.results[key] = {"result": result, "audio_id": audio_id, "rendered": None}
        return key

    def job_snapshot(self, key):
        job = self.lookup(self.jobs, key)
        with self.lock:
            return {name: value for name, value in job.items() if name not in ("cancel", "future")}

    def submit(self, kind, operation):
        key = secrets.token_hex(12)
        job = {"id": key, "kind": kind, "status": "queued", "completed": 0, "total": 0,
               "cancel": threading.Event()}
        with self.lock:
            # A local app should not accumulate many expensive queued GPU tasks.
            if any(item["status"] in ("queued", "running") for item in self.jobs.values()):
                raise ValueError("A task is already running. Finish it or cancel it first.")
            self.jobs[key] = job

        def progress(completed, total):
            with self.lock:
                job.update(completed=completed, total=total)

        def run():
            try:
                if job["cancel"].is_set():
                    raise InterruptedError("Task cancelled.")
                with self.lock:
                    job["status"] = "running"
                payload = operation(progress, job["cancel"])
                if job["cancel"].is_set():
                    raise InterruptedError("Task cancelled.")
                with self.lock:
                    job.update(status="complete", **payload)
            except InterruptedError:
                with self.lock:
                    job["status"] = "cancelled"
            except Exception as error:
                with self.lock:
                    job.update(status="error", error=str(error))
        job["future"] = self.executor.submit(run)
        return {"id": key}

    def analyze(self, options):
        audio = self.lookup(self.audio, options.get("audio_id"))
        model = self.lookup(self.models, options.get("model_id"))
        device, mode = options.get("device", "auto"), options.get("overlap_blend", "weighted")
        overlap = float(options.get("overlap_seconds", 2))
        if device not in ("auto", "cpu", "cuda") or mode not in ("crop", "weighted", "equal"):
            raise ValueError("Invalid device or overlap method.")
        if not math.isfinite(overlap) or overlap not in (0, 1, 2, 3, 4):
            raise ValueError("Choose overlap from 0 to 4 seconds.")
        overrides = {}
        if not options.get("use_model_thresholds", True):
            for head in ("frame", "onset", "offset"):
                value = float(options[head + "_threshold"])
                if not math.isfinite(value) or not 0 < value < 1:
                    raise ValueError("Thresholds must be between zero and one.")
                overrides[head + "_threshold"] = value

        def operation(progress, cancel):
            result = transcribe_audio(audio["path"], model, device=device, progress=progress, cancel=cancel,
                                      context_seconds=overlap / 2, overlap_blend=mode, **overrides)
            result["audio"] = audio["name"]
            result["model"] = self.model_names.get(options.get("model_id"), model.name)
            return {"result_id": self.add_result(result, audio["id"])}
        return self.submit("analysis", operation)

    def render_audio(self, key):
        item = self.lookup(self.results, key)
        if item["rendered"] and item["rendered"].is_file():
            return {"url": f"/api/results/{key}/audio"}
        def operation(progress, cancel):
            path = self.directory / f"{key}-piano.wav"
            render_result_wav(item["result"], path, progress=progress, cancel=cancel)
            with self.lock:
                item["rendered"] = path
            return {"url": f"/api/results/{key}/audio"}
        return self.submit("piano_render", operation)

    def view(self, key, query, png=False):
        item = self.lookup(self.results, key)
        result = item["result"]
        kind = query.get("view", ["staff"])[0]
        if kind not in ("staff", "timeline"):
            raise ValueError("Unknown view.")
        with self.plot_lock:
            figure = Figure(figsize=(12, 7), dpi=120)
            if kind == "staff":
                bpm = float(query.get("bpm", [120])[0])
                page = int(query.get("page", [0])[0])
                if not math.isfinite(bpm):
                    raise ValueError("Invalid tempo.")
                axis, page, pages, window = draw_staff(figure, result, bpm, page,
                                                      names=query.get("names", ["false"])[0] == "true")
                meta = {"page": page, "pages": pages, "start": window[0], "end": window[1], "bpm": bpm}
            else:
                envelope = None
                if item["audio_id"]:
                    envelope = waveform_envelope(self.lookup(self.audio, item["audio_id"])["path"])
                axes = draw_prediction(figure, result, envelope)
                start = float(query.get("start", [0])[0])
                span = float(query.get("span", [result["duration"]])[0])
                if not math.isfinite(start + span) or start < 0 or span <= 0:
                    raise ValueError("Timeline start must be nonnegative and span must be positive.")
                span = min(span, result["duration"])
                start = min(start, max(0, result["duration"] - span))
                end = min(result["duration"], start + span)
                for axis in axes:
                    axis.set_xlim(start, end)
                axis = axes[1]
                meta = {"start": start, "end": end}
            meta.update(bounds=list(axis.get_position().bounds), xlim=list(axis.get_xlim()))
            buffer = io.BytesIO()
            figure.savefig(buffer, format="png" if png else "svg")
            figure.clear()
            if png:
                return buffer.getvalue()
            return {**meta, "svg": buffer.getvalue().decode("utf-8")}


class PianoHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, state):
        self.state = state
        super().__init__(address, PianoHandler)


class PianoHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def handle(self):
        try:
            super().handle()
        except ConnectionError:
            # Browsers may abort a media Range request when pausing or seeking.
            pass

    def log_message(self, format, *args):
        # Frequent progress polls would obscure startup and errors in the terminal.
        if args and str(args[1] if len(args) > 1 else "").startswith("5"):
            super().log_message(format, *args)

    @property
    def state(self):
        return self.server.state

    def headers_out(self, status, content_type, length, extra=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        origin = self.headers.get("Origin")
        if origin in self.state.site_origins:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Expose-Headers", "Content-Range, Content-Disposition, Accept-Ranges")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' blob:; media-src 'self'; connect-src 'self'; frame-ancestors 'none'")
        for name, value in (extra or {}).items():
            self.send_header(name, value)
        self.end_headers()

    def respond(self, data, status=200):
        encoded = json.dumps(data, allow_nan=False).encode("utf-8")
        self.headers_out(status, "application/json; charset=utf-8", len(encoded))
        self.wfile.write(encoded)

    def validate_request(self, mutation=False):
        authority = self.headers.get("Host", "")
        allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        if authority not in allowed:
            raise PermissionError("Open the app through its localhost URL.")
        origin = self.headers.get("Origin")
        if origin and origin not in {"http://" + host for host in allowed} | self.state.site_origins:
            raise PermissionError("This website is not allowed to connect to the piano engine.")
        if mutation and not secrets.compare_digest(self.headers.get("X-Piano-Token", ""), self.state.token):
            raise PermissionError("Reload the page before uploading or starting a task.")

    def request_path(self):
        parsed = urlsplit(self.path)
        return parsed.path, parse_qs(parsed.query)

    def read_body(self, maximum):
        length = int(self.headers.get("Content-Length", "0"))
        if not 0 < length <= maximum:
            raise ValueError(f"Choose a file smaller than {maximum // (1024 * 1024)} MB.")
        return length

    def read_json(self):
        length = self.read_body(2 * 1024 * 1024)
        payload = json.loads(self.rfile.read(length))
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON object.")
        return payload

    def receive_file(self, suffix, maximum):
        length = self.read_body(maximum)
        path = self.state.directory / (secrets.token_hex(12) + suffix)
        try:
            with path.open("wb") as output:
                while length:
                    block = self.rfile.read(min(length, 1024 * 1024))
                    if not block:
                        raise ValueError("Upload interrupted.")
                    output.write(block)
                    length -= len(block)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        name = Path(unquote(self.headers.get("X-Filename", "upload"))).name.replace("\r", "").replace("\n", "")
        return path, name

    def file_response(self, path, content_type=None, download=None):
        size = path.stat().st_size
        first, last, status = 0, size - 1, 200
        extra = {"Accept-Ranges": "bytes"}
        if download:
            extra["Content-Disposition"] = f'attachment; filename="{download}"'
        range_header = self.headers.get("Range")
        if range_header:
            try:
                if not range_header.startswith("bytes=") or "," in range_header:
                    raise ValueError
                left, right = range_header[6:].split("-", 1)
                if left:
                    first, last = int(left), min(size - 1, int(right)) if right else size - 1
                else:
                    suffix = int(right)
                    if suffix <= 0:
                        raise ValueError
                    first = max(0, size - suffix)
                if not 0 <= first <= last < size:
                    raise ValueError
            except (ValueError, TypeError):
                self.headers_out(416, "text/plain", 0, {"Content-Range": f"bytes */{size}"})
                return
            status = 206
            extra["Content-Range"] = f"bytes {first}-{last}/{size}"
        count = last - first + 1
        self.headers_out(status, content_type or mimetypes.guess_type(path)[0] or "application/octet-stream", count, extra)
        with path.open("rb") as source:
            source.seek(first)
            while count:
                block = source.read(min(count, 128 * 1024))
                if not block:
                    break
                self.wfile.write(block)
                count -= len(block)

    def do_GET(self):
        self.dispatch(False)

    def do_POST(self):
        self.dispatch(True)

    def do_OPTIONS(self):
        try:
            self.validate_request()
            if self.headers.get("Access-Control-Request-Method") not in {"GET", "POST"}:
                raise PermissionError("Unsupported request method.")
            requested = {name.strip().lower() for name in self.headers.get("Access-Control-Request-Headers", "").split(",") if name.strip()}
            if not requested <= {"content-type", "x-piano-token", "x-filename", "range"}:
                raise PermissionError("Unsupported request headers.")
            extra = {"Access-Control-Allow-Methods": "GET, POST", "Access-Control-Allow-Headers": "Content-Type, X-Piano-Token, X-Filename, Range", "Access-Control-Max-Age": "300"}
            if self.headers.get("Origin") in self.state.site_origins:
                extra["Access-Control-Allow-Private-Network"] = "true"
            self.headers_out(204, "text/plain", 0, extra)
        except PermissionError as error:
            self.respond({"error": str(error)}, 403)

    def dispatch(self, mutation):
        try:
            self.validate_request(mutation)
            path, query = self.request_path()
            parts = path.strip("/").split("/")
            if not mutation:
                if path in ("/", "/app.css", "/app.js", "/transport.js"):
                    content_type = "text/html; charset=utf-8" if path == "/" else "text/css; charset=utf-8" if path == "/app.css" else "text/javascript; charset=utf-8"
                    return self.file_response(STATIC / ("index.html" if path == "/" else path[1:]), content_type)
                if path == "/api/bootstrap":
                    return self.respond({"token": self.state.token, "models": self.state.refresh_models(),
                                         "cuda": torch.cuda.is_available(), "piano_sound": piano_sound_name()})
                if path == "/api/models":
                    return self.respond({"models": self.state.refresh_models()})
                if len(parts) == 3 and parts[:2] == ["api", "models"]:
                    return self.respond(self.state.model_details(parts[2]))
                if len(parts) == 3 and parts[:2] == ["api", "jobs"]:
                    return self.respond(self.state.job_snapshot(parts[2]))
                if len(parts) == 4 and parts[:2] == ["api", "audio"] and parts[3] == "file":
                    return self.file_response(self.state.lookup(self.state.audio, parts[2])["path"], "audio/wav")
                if len(parts) in (3, 4) and parts[:2] == ["api", "results"]:
                    key, action = parts[2], parts[3] if len(parts) == 4 else "result"
                    item = self.state.lookup(self.state.results, key)
                    if action == "result":
                        return self.respond({"id": key, "result": item["result"], "audio_id": item["audio_id"]})
                    if action == "view":
                        return self.respond(self.state.view(key, query))
                    if action == "view.png":
                        data = self.state.view(key, query, png=True)
                        self.headers_out(200, "image/png", len(data), {"Content-Disposition": 'attachment; filename="piano-view.png"'})
                        return self.wfile.write(data)
                    if action == "export":
                        data = json.dumps(item["result"], indent=2).encode("utf-8")
                        self.headers_out(200, "application/json", len(data), {"Content-Disposition": 'attachment; filename="prediction.json"'})
                        return self.wfile.write(data)
                    if action == "audio":
                        if not item["rendered"]:
                            raise ValueError("Choose Play result to prepare the piano audio first.")
                        return self.file_response(item["rendered"], "audio/wav",
                                                  "piano-result.wav" if "download" in query else None)
            else:
                if path in ("/api/audio", "/api/models/upload", "/api/results/import"):
                    suffix, limit = {"/api/audio": (".wav", 2 * 1024 ** 3),
                                     "/api/models/upload": (".pt", 512 * 1024 ** 2),
                                     "/api/results/import": (".json", 32 * 1024 ** 2)}[path]
                    file, name = self.receive_file(suffix, limit)
                    try:
                        if path == "/api/audio":
                            return self.respond(self.state.register_audio(file, name), 201)
                        if path == "/api/models/upload":
                            key = secrets.token_hex(12)
                            with self.state.lock:
                                self.state.models[key] = file
                                self.state.model_names[key] = name
                            try:
                                details = self.state.model_details(key)
                            except BaseException:
                                with self.state.lock:
                                    self.state.models.pop(key, None)
                                    self.state.model_names.pop(key, None)
                                raise
                            details["name"] = name
                            return self.respond(details, 201)
                        result = read_prediction(file)
                        key = self.state.add_result(result)
                        return self.respond({"id": key, "result": result, "audio_id": None}, 201)
                    except BaseException:
                        file.unlink(missing_ok=True)
                        raise
                if path == "/api/analyze":
                    return self.respond(self.state.analyze(self.read_json()), 202)
                if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "cancel":
                    if int(self.headers.get("Content-Length", "0")):
                        self.read_json()
                    self.state.lookup(self.state.jobs, parts[2])["cancel"].set()
                    return self.respond({"cancelled": True})
                if len(parts) == 4 and parts[:2] == ["api", "results"] and parts[3] == "render":
                    if int(self.headers.get("Content-Length", "0")):
                        self.read_json()
                    return self.respond(self.state.render_audio(parts[2]), 202)
            self.respond({"error": "Not found."}, 404)
        except ConnectionError:
            pass
        except PermissionError as error:
            self.close_connection = True
            self.respond({"error": str(error)}, 403)
        except KeyError as error:
            self.close_connection = True
            self.respond({"error": str(error)}, 404)
        except (ValueError, TypeError, OSError, wave.Error) as error:
            self.close_connection = True
            self.respond({"error": str(error)}, 400)
        except Exception as error:
            self.close_connection = True
            self.respond({"error": str(error)}, 500)


def main():
    parser = argparse.ArgumentParser(description="Run the local piano web app.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="Print the URL without opening a browser")
    parser.add_argument("--sites", action="store_true", help="Open the browser-only Site without starting a local engine")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("Port must be between 0 and 65535.")
    site_url = None
    if args.sites:
        try:
            site_url = json.loads((PROJECT / "piano-site/site.json").read_text(encoding="utf-8"))["url"]
            parsed = urlsplit(site_url)
            if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
                raise ValueError("Expected an HTTPS Site URL.")
            site_origin = f"{parsed.scheme}://{parsed.netloc}"
        except (OSError, KeyError, ValueError) as error:
            parser.exit(1, f"Cannot find the published Site configuration: {error}\n")
    if site_url:
        print(f"Piano Notes Site: {site_url}\nChoose your exported ONNX model and WAV directly in the Site. No local engine is needed.", flush=True)
        if not args.no_browser:
            webbrowser.open(site_url)
        return
    state = WebState()
    try:
        server = PianoHTTPServer(("127.0.0.1", args.port), state)
    except OSError as error:
        state.close()
        parser.exit(1, f"Cannot start the web app: {error}. Try --port 8766.\n")
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"Piano Notes web app: {url}\nKeep this terminal open. Press Ctrl+C to stop.", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        state.close()


if __name__ == "__main__":
    main()
