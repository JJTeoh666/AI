import io
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch
import wave

import numpy as np
import torch

from piano_ml.model import FourierRecurrentPianoNet, ThreeBranchPianoNet, HarmonicCQTPianoNet, STFTCQTPianoNet
from piano_ml.synthesis import render_result_wav
from piano_ml.web import PianoHTTPServer, WebState


class WebAppTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.project = Path(self.temp.name)
        (self.project / "checkpoints").mkdir()
        model = ThreeBranchPianoNet(feature_width=16, hidden_size=8, gru_layers=1)
        torch.save({"model": model.state_dict(), "model_config": model.config, "model_version": "9"},
                   self.project / "checkpoints/piano-v9.pt")
        v81 = FourierRecurrentPianoNet(feature_width=16, hidden_size=8, gru_layers=1, fourier_layers=1)
        torch.save({"model": v81.state_dict(), "model_config": v81.config, "model_version": "8.1"},
                   self.project / "checkpoints/piano-v8.1.pt")
        self.site_origin = "https://piano.example.test"
        self.state = WebState(self.project, site_origins=(self.site_origin,))
        self.server = PianoHTTPServer(("127.0.0.1", 0), self.state)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        self.token = self.request("/api/bootstrap")["token"]

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.state.close()
        self.temp.cleanup()

    def request(self, path, body=None, method="GET", headers=None, raw=False):
        options = {"X-Piano-Token": self.state.token, **(headers or {})}
        if isinstance(body, dict):
            body = json.dumps(body).encode()
            options["Content-Type"] = "application/json"
        with urlopen(Request(self.base + path, data=body, headers=options, method=method), timeout=15) as response:
            content = response.read()
            return (response.status, dict(response.headers), content) if raw else json.loads(content)

    def wav_bytes(self, seconds=1):
        output = io.BytesIO()
        with wave.open(output, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            samples = np.sin(np.arange(round(seconds * 16000)) * (2 * np.pi * 440 / 16000)) * 6000
            handle.writeframes(samples.astype("<i2").tobytes())
        return output.getvalue()

    def audio(self):
        return self.request("/api/audio", self.wav_bytes(), "POST", {"X-Filename": "piano.wav"})

    @staticmethod
    def result():
        return {"duration": 1.0, "notes": [{"pitch": 60, "start": 0.1, "end": 0.7,
                "confidence": 0.9, "velocity": 90}], "chords": [{"name": "C", "start": 0.1, "end": 0.7}],
                "pedals": [{"start": 0.2, "end": 0.9}]}

    def wait(self, job):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            result = self.request("/api/jobs/" + job["id"])
            if result["status"] in ("complete", "cancelled", "error"):
                return result
            time.sleep(0.02)
        self.fail("Web task did not finish")

    def test_static_bootstrap_and_model_version(self):
        bootstrap = self.request("/api/bootstrap")
        self.assertEqual(bootstrap["models"][0]["name"], "piano-v8.1.pt")
        info = self.request("/api/models/" + bootstrap["models"][0]["id"])
        self.assertEqual(info["version"], "8.1")
        self.assertEqual(set(info["thresholds"]), {"frame", "onset", "offset"})
        for path, kind in (("/", "text/html"), ("/app.js", "text/javascript"), ("/transport.js", "text/javascript"), ("/app.css", "text/css")):
            code, headers, content = self.request(path, raw=True)
            self.assertEqual(code, 200)
            self.assertTrue(headers["Content-Type"].startswith(kind))
            self.assertGreater(len(content), 100)
        self.assertIn(b'aria-label="Playback position; drag to seek"', self.request("/", raw=True)[2])

    def test_original_audio_supports_browser_range_seeking(self):
        data = self.wav_bytes()
        audio = self.request("/api/audio", data, "POST", {"X-Filename": "piano.wav"})
        for value, first, last in (("bytes=0-43", 0, 43), ("bytes=100-299", 100, 299),
                                   ("bytes=-32", len(data) - 32, len(data) - 1),
                                   ("bytes=200-", 200, len(data) - 1)):
            with self.subTest(range=value):
                code, headers, body = self.request(audio["url"], headers={"Range": value}, raw=True)
                self.assertEqual(code, 206)
                self.assertEqual(body, data[first:last + 1])
                self.assertEqual(headers["Content-Range"], f"bytes {first}-{last}/{len(data)}")
        self.assertEqual(self.request(audio["url"], raw=True)[2], data)
        for value in ("bytes=999999-", "bytes=-0", "bytes=50-10", "bytes=0-1,4-5"):
            with self.assertRaises(HTTPError) as error:
                self.request(audio["url"], headers={"Range": value}, raw=True)
            self.assertEqual(error.exception.code, 416)

    def test_analysis_worker_reuses_thresholds_overlap_and_device(self):
        audio = self.audio()
        model = self.request("/api/models")["models"][0]
        def transcribe(*args, progress, cancel, **kwargs):
            self.assertEqual(kwargs, {"device": "cpu", "context_seconds": 1.5, "overlap_blend": "equal",
                                      "frame_threshold": 0.31, "onset_threshold": 0.42, "offset_threshold": 0.63})
            progress(1, 1)
            return self.result()
        with patch("piano_ml.web.transcribe_audio", side_effect=transcribe):
            job = self.request("/api/analyze", {"audio_id": audio["id"], "model_id": model["id"],
                    "device": "cpu", "overlap_seconds": 3, "overlap_blend": "equal", "use_model_thresholds": False,
                    "frame_threshold": 0.31, "onset_threshold": 0.42, "offset_threshold": 0.63}, "POST")
            complete = self.wait(job)
        self.assertEqual(complete["status"], "complete")
        self.assertEqual(complete["completed"], 1)
        result = self.request("/api/results/" + complete["result_id"])
        self.assertEqual(result["audio_id"], audio["id"])
        self.assertEqual(result["result"]["audio"], "piano.wav")

    def test_real_v9_analysis_all_blend_methods(self):
        audio = self.audio()
        model = next(item for item in self.request("/api/models")["models"] if item["name"] == "piano-v9.pt")
        for mode in ("crop", "weighted", "equal"):
            job = self.request("/api/analyze", {"audio_id": audio["id"], "model_id": model["id"],
                               "device": "cpu", "overlap_blend": mode}, "POST")
            complete = self.wait(job)
            self.assertEqual(complete["status"], "complete", complete)
            result = self.request("/api/results/" + complete["result_id"])["result"]
            self.assertEqual(result["model_version"], "9")
            self.assertEqual(result["inference_config"]["overlap_blend"], mode)
            self.assertEqual(result["inference_config"]["overlap_seconds"], 2)

    def test_site_cors_preflight_upload_and_media(self):
        headers = {"Origin": self.site_origin}
        code, response_headers, content = self.request("/api/bootstrap", headers=headers, raw=True)
        self.assertEqual(response_headers["Access-Control-Allow-Origin"], self.site_origin)
        self.assertEqual(json.loads(content)["models"][0]["name"], "piano-v8.1.pt")
        code, response_headers, _ = self.request("/api/audio", method="OPTIONS", headers={**headers,
            "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "x-piano-token,x-filename,content-type",
            "Access-Control-Request-Private-Network": "true"}, raw=True)
        self.assertEqual(code, 204)
        self.assertEqual(response_headers["Access-Control-Allow-Private-Network"], "true")
        audio = self.request("/api/audio", self.wav_bytes(), "POST", headers)
        code, response_headers, data = self.request(audio["url"], headers={**headers, "Range": "bytes=0-43"}, raw=True)
        self.assertEqual(code, 206)
        self.assertEqual(response_headers["Access-Control-Allow-Origin"], self.site_origin)
        self.assertEqual(data[:4], b"RIFF")
        for method in ("GET", "OPTIONS"):
            with self.assertRaises(HTTPError) as error:
                self.request("/api/bootstrap", method=method, headers={"Origin": "https://other.example.test",
                    "Access-Control-Request-Method": "POST"})
            self.assertEqual(error.exception.code, 403)
            self.assertIsNone(error.exception.headers.get("Access-Control-Allow-Origin"))
        with self.assertRaises(HTTPError) as error:
            self.request("/api/audio", self.wav_bytes(), "POST", {**headers, "X-Piano-Token": "wrong"})
        self.assertEqual(error.exception.code, 403)

    def test_saved_result_views_piano_audio_and_exports(self):
        saved = self.request("/api/results/import", json.dumps(self.result()).encode(), "POST")
        key = saved["id"]
        self.assertEqual(saved["result"]["notes"][0]["name"], "C4")
        for kind in ("staff", "timeline"):
            view = self.request(f"/api/results/{key}/view?view={kind}&names=true&bpm=120")
            self.assertIn("<svg", view["svg"])
            self.assertEqual(len(view["bounds"]), 4)
            self.assertLess(view["xlim"][0], view["xlim"][1])
        zoomed = self.request(f"/api/results/{key}/view?view=timeline&start=0.25&span=0.25")
        self.assertEqual(zoomed["start"], 0.25)
        self.assertEqual(zoomed["end"], 0.5)
        self.assertEqual(zoomed["xlim"], [0.25, 0.5])
        self.assertTrue(self.request(f"/api/results/{key}/view.png?view=staff", raw=True)[2].startswith(b"\x89PNG"))
        self.assertEqual(self.request(f"/api/results/{key}/export")["notes"][0]["name"], "C4")
        with patch("piano_ml.web.render_result_wav", side_effect=lambda *a, **kw: render_result_wav(*a, engine="basic", **kw)):
            complete = self.wait(self.request(f"/api/results/{key}/render", {}, "POST"))
        self.assertEqual(complete["status"], "complete", complete)
        code, headers, body = self.request(complete["url"] + "?download=1", raw=True)
        self.assertEqual(code, 200)
        self.assertIn("attachment", headers["Content-Disposition"])
        with wave.open(io.BytesIO(body), "rb") as handle:
            self.assertGreater(handle.getnframes(), 16000)
        self.assertEqual(self.request(f"/api/results/{key}/render", {}, "POST")["url"], complete["url"])

    def test_checkpoint_browse_keeps_display_name(self):
        source = self.project / "checkpoints/piano-v9.pt"
        uploaded = self.request("/api/models/upload", source.read_bytes(), "POST", {"X-Filename": "experiment-v9.pt"})
        self.assertEqual(uploaded["name"], "experiment-v9.pt")
        row = next(item for item in self.request("/api/models")["models"] if item["id"] == uploaded["id"])
        self.assertEqual(row["name"], "experiment-v9.pt")

    def test_v10_checkpoint_upload_metadata_and_analysis(self):
        model = HarmonicCQTPianoNet(feature_width=16, hidden_size=8, gru_layers=1)
        source = self.project / "v10.pt"
        torch.save({"model": model.state_dict(), "model_config": model.config, "model_version": "10"}, source)
        uploaded = self.request("/api/models/upload", source.read_bytes(), "POST", {"X-Filename": "experiment-v10.pt"})
        self.assertEqual(uploaded["version"], "10")
        self.assertEqual(uploaded["parameters"], sum(p.numel() for p in model.parameters()))
        self.assertEqual(self.request("/api/models")["models"][0]["name"], "piano-v8.1.pt")
        audio = self.audio()
        job = self.request("/api/analyze", {"audio_id": audio["id"], "model_id": uploaded["id"],
                          "device": "cpu", "overlap_blend": "weighted"}, "POST")
        complete = self.wait(job)
        self.assertEqual(complete["status"], "complete", complete)
        result = self.request("/api/results/" + complete["result_id"])["result"]
        self.assertEqual(result["model_version"], "10")
        self.assertEqual(result["architecture"], model.architecture)
        self.assertEqual(result["inference_config"]["overlap_blend"], "weighted")

    def test_v101_checkpoint_upload_and_analysis_preserve_web_default(self):
        model = STFTCQTPianoNet(feature_width=16, hidden_size=8, gru_layers=1)
        source = self.project / "v101.pt"
        torch.save({"model": model.state_dict(), "model_config": model.config}, source)
        uploaded = self.request("/api/models/upload", source.read_bytes(), "POST", {"X-Filename": "experiment-v10.1.pt"})
        self.assertEqual(uploaded["version"], "10.1")
        self.assertEqual(self.request("/api/models")["models"][0]["name"], "piano-v8.1.pt")
        audio = self.audio()
        job = self.request("/api/analyze", {"audio_id": audio["id"], "model_id": uploaded["id"],
                          "device": "cpu", "overlap_blend": "weighted"}, "POST")
        complete = self.wait(job)
        self.assertEqual(complete["status"], "complete", complete)
        result = self.request("/api/results/" + complete["result_id"])["result"]
        self.assertEqual(result["model_version"], "10.1")
        self.assertEqual(result["architecture"], model.architecture)

    def test_cancellation_and_second_task_rejection(self):
        audio = self.audio()
        model = self.request("/api/models")["models"][0]
        started = threading.Event()
        def blocked(*args, cancel, **kwargs):
            started.set()
            cancel.wait(10)
            raise InterruptedError
        options = {"audio_id": audio["id"], "model_id": model["id"]}
        with patch("piano_ml.web.transcribe_audio", side_effect=blocked):
            job = self.request("/api/analyze", options, "POST")
            self.assertTrue(started.wait(3))
            with self.assertRaises(HTTPError) as error:
                self.request("/api/analyze", options, "POST")
            self.assertEqual(error.exception.code, 400)
            self.request(f"/api/jobs/{job['id']}/cancel", {}, "POST")
            self.assertEqual(self.wait(job)["status"], "cancelled")

    def test_invalid_files_controls_and_local_request_checks(self):
        for path, body in (("/api/audio", b"not a WAV"), ("/api/models/upload", b"not a checkpoint"),
                           ("/api/results/import", b'{"duration":-1}')):
            with self.assertRaises(HTTPError) as error:
                self.request(path, body, "POST")
            self.assertIn(error.exception.code, (400, 500))
        audio = self.audio()
        model = self.request("/api/models")["models"][0]
        options = {"audio_id": audio["id"], "model_id": model["id"], "overlap_seconds": 8}
        with self.assertRaises(HTTPError) as error:
            self.request("/api/analyze", options, "POST")
        self.assertEqual(error.exception.code, 400)
        for headers in ({"X-Piano-Token": "wrong"}, {"Origin": "https://example.com"}, {"Host": "untrusted.example"}):
            with self.assertRaises(HTTPError) as error:
                self.request("/api/audio", self.wav_bytes(), "POST", headers)
            self.assertEqual(error.exception.code, 403)
        with self.assertRaises(HTTPError) as error:
            self.request("/../README.md")
        self.assertEqual(error.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
