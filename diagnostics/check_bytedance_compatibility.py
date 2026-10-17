"""Check CPU/CUDA inference; optionally compare with pinned upstream calculations.

Run from the project root. --reference downloads source into ignored .deps/.
It does not install packages or change the environment.
"""

import argparse
import ast
import copy
import json
import platform
import sys
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from piano_ml.bytedance import (SOURCE_COMMIT, _load_model, read_audio, transcribe_bytedance)


def source(path, repo="bytedance/piano_transcription", commit=SOURCE_COMMIT):
    directory = ROOT / ".deps/bytedance-reference"
    directory.mkdir(parents=True, exist_ok=True)
    cached = directory / (repo.replace("/", "-") + "-" + commit[:12] + "-" + path.replace("/", "_"))
    if not cached.is_file():
        url = f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"
        request = urllib.request.Request(url, headers={"User-Agent": "PianoNotesStudio"})
        cached.write_bytes(urllib.request.urlopen(request, timeout=60).read())
    return cached.read_text(encoding="utf-8")


def reference_check():
    # Execute upstream forward methods with checkpoint-restored frontend tensors.
    # Constructors are bypassed to avoid installing librosa just to create DFTs.
    frontend = source("torchlibrosa/stft.py", "qiuqiangkong/torchlibrosa", "d4e8add6c9ee49ae531c78a4901e4fdcc88c746a")
    tree = ast.parse(frontend)
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name in ("DFTBase", "STFT", "Spectrogram", "LogmelFilterBank")]
    namespace = {"torch": torch, "nn": nn, "F": F, "np": np}
    exec(compile(ast.Module(body=classes, type_ignores=[]), "upstream-frontend", "exec"), namespace)

    def spectrogram(**kwargs):
        stft = namespace["STFT"].__new__(namespace["STFT"])
        nn.Module.__init__(stft)
        stft.n_fft, stft.center, stft.pad_mode = 2048, True, "reflect"
        stft.conv_real = nn.Conv1d(1, 1025, 2048, stride=160, bias=False)
        stft.conv_imag = nn.Conv1d(1, 1025, 2048, stride=160, bias=False)
        result = namespace["Spectrogram"].__new__(namespace["Spectrogram"])
        nn.Module.__init__(result)
        result.stft, result.power = stft, 2.
        return result

    def logmel(**kwargs):
        result = namespace["LogmelFilterBank"].__new__(namespace["LogmelFilterBank"])
        nn.Module.__init__(result)
        result.melW = nn.Parameter(torch.empty(1025, 229), requires_grad=False)
        result.is_log, result.ref, result.amin, result.top_db = True, 1., 1e-10, None
        return result

    model_source = source("pytorch/models.py")
    model_source = model_source.replace("from torchlibrosa.stft import Spectrogram, LogmelFilterBank", "")
    model_source = model_source.replace("from pytorch_utils import move_data_to_device", "")
    namespace_model = {"Spectrogram": spectrogram, "LogmelFilterBank": logmel}
    exec(compile(model_source, "upstream-models", "exec"), namespace_model)
    with torch.device("meta"):
        reference = namespace_model["Note_pedal"](100, 88)
    model = _load_model("cpu")
    reference.note_model.load_state_dict(model.note_model.state_dict(), strict=True, assign=True)
    reference.pedal_model.load_state_dict(model.pedal_model.state_dict(), strict=True, assign=True)
    reference.eval()
    samples, _ = read_audio(ROOT / "diagnostics/browser-clip.wav")
    wave = torch.from_numpy(np.pad(samples, (0, 160000 - len(samples))))[None]
    with torch.inference_mode():
        actual, expected = model(wave), reference(wave)
    differences = {key: float((actual[key] - expected[key]).abs().max()) for key in actual}
    # Float32 kernels can differ slightly with parameter layout / GRU backend.
    assert all(value < 1e-5 for value in differences.values()), differences

    decoder_source = source("utils/utilities.py")
    decoder_class = next(node for node in ast.parse(decoder_source).body
                         if isinstance(node, ast.ClassDef) and node.name == "RegressionPostProcessor")
    namespace_decoder = {"np": np, "config": SimpleNamespace(begin_note=21, velocity_scale=128)}
    exec(compile(source("utils/piano_vad.py"), "upstream-piano-vad", "exec"), namespace_decoder)
    exec(compile(ast.Module(body=[decoder_class], type_ignores=[]), "upstream-decoder", "exec"), namespace_decoder)
    upstream_decoder = namespace_decoder["RegressionPostProcessor"](100, 88, .3, .3, .1, .2)
    from piano_ml._bytedance.decoder import RegressionPostProcessor
    local_decoder = RegressionPostProcessor(100, 88, .3, .3, .1, .2)
    values = {key: value[0].numpy() for key, value in expected.items()}
    expected_events = upstream_decoder.output_dict_to_midi_events(copy.deepcopy(values))
    actual_events = local_decoder.output_dict_to_midi_events(copy.deepcopy(values))
    assert actual_events == expected_events
    return {"all_head_max_abs_difference": differences, "absolute_tolerance": 1e-5,
            "upstream_decoder_events_identical": True,
            "note_events": len(actual_events[0]), "pedal_events": len(actual_events[1])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = {"python": platform.python_version(), "torch": str(torch.__version__), "devices": {}}
    for device in ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",):
        result = transcribe_bytedance(ROOT / "diagnostics/browser-clip.wav", device=device)
        report["devices"][device] = {"notes": len(result["notes"]), "pedals": len(result["pedals"]),
                                      "analysis_seconds": result["analysis_seconds"]}
        filename = "bytedance-desktop-result.json" if device == "cpu" else "bytedance-cuda-result.json"
        (ROOT / "diagnostics" / filename).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(device, report["devices"][device], flush=True)
    _load_model.cache_clear()
    if args.reference:
        report["reference"] = reference_check()
    (ROOT / "diagnostics/bytedance-compatibility.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
