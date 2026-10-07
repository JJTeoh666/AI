import tempfile
import threading
import unittest
import wave
from itertools import product
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from piano_ml.decode import decode_outputs
from piano_ml.inference import predict_probabilities, transcribe_audio
from piano_ml.model import FourierRecurrentPianoNet, HOP_LENGTH, SAMPLE_RATE


class AudioMarkerModel:
    """Expose audio frame markers so stitching can be checked against the whole WAV."""
    architecture = "onsets"
    config = {"architecture": "onsets"}

    def __init__(self, architecture="onsets"):
        self.architecture = architecture
        self.config = {"architecture": architecture}
        self.inputs = []

    def __call__(self, samples):
        self.inputs.append(samples.clone())
        markers = samples[:, ::HOP_LENGTH]
        frames = samples.shape[1] // HOP_LENGTH + 1
        markers = torch.nn.functional.pad(markers, (0, frames - markers.shape[1]))
        logits = markers.unsqueeze(-1)
        if self.architecture == "frame":
            return logits.expand(-1, -1, 88)
        return {**{head: logits.expand(-1, -1, 88)
                   for head in ("frame", "onset", "offset", "velocity")}, "pedal": logits}


class InferenceOverlapTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.root = Path(self.temp.name)
        self.audio = self.root / "markers.wav"
        # Distinct frame markers, spanning two boundaries and a short final section.
        self.samples = np.repeat(np.arange(402, dtype="<i2"), HOP_LENGTH)
        with wave.open(str(self.audio), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(self.samples.tobytes())
        self.expected = torch.from_numpy(self.samples[::HOP_LENGTH].astype(np.float32) / 32768).sigmoid().numpy()

    def tearDown(self):
        self.temp.cleanup()

    def test_all_overlap_choices_preserve_every_head_and_timestamps(self):
        for overlap, blend in product((0, 1, 2, 3, 4), ("crop", "weighted", "equal")):
            with self.subTest(overlap=overlap, blend=blend):
                model = AudioMarkerModel()
                progress = []
                duration, output = predict_probabilities(
                    self.audio, model, torch.device("cpu"),
                    progress=lambda done, total: progress.append((done, total)),
                    context_seconds=overlap / 2, overlap_blend=blend)
                self.assertEqual(duration, 8.04)
                self.assertEqual(progress, [(1, 3), (2, 3), (3, 3)])
                self.assertEqual(model.inputs[0].shape[1], round((4 + overlap / 2) * SAMPLE_RATE))
                self.assertEqual(model.inputs[1].shape[1], (4 + overlap) * SAMPLE_RATE)
                self.assertEqual(model.inputs[-1][0, -1].item(), 0)  # end padding
                for head, probabilities in output.items():
                    width = 1 if head == "pedal" else 88
                    self.assertEqual(probabilities.shape, (402, width))
                    expected = np.repeat(self.expected[:, None], width, axis=1)
                    if blend == "crop" or overlap == 0:
                        np.testing.assert_array_equal(probabilities, expected)
                    else:
                        np.testing.assert_allclose(probabilities, expected, rtol=0, atol=1e-7)

    def test_legacy_defaults_and_explicit_context(self):
        for architecture, default_context in (("frame", 0), ("onsets", 0.5)):
            with self.subTest(architecture=architecture):
                model = AudioMarkerModel(architecture)
                _, output = predict_probabilities(self.audio, model, torch.device("cpu"))
                self.assertEqual(model.inputs[0].shape[1], round((4 + default_context) * SAMPLE_RATE))
                np.testing.assert_array_equal(output["frame"][:, 0], self.expected)
        for blend in ("weighted", "equal"):
            with self.subTest(blend=blend):
                legacy = AudioMarkerModel("frame")
                _, output = predict_probabilities(self.audio, legacy, torch.device("cpu"), context_seconds=1,
                                                 overlap_blend=blend)
                np.testing.assert_allclose(output["frame"][:, 0], self.expected, rtol=0, atol=1e-7)
                self.assertEqual(legacy.inputs[1].shape[1], 6 * SAMPLE_RATE)

    def test_invalid_context_and_cancellation(self):
        model = AudioMarkerModel()
        for context in (-1, 2.1, float("nan"), float("inf")):
            with self.subTest(context=context), self.assertRaisesRegex(ValueError, "context"):
                predict_probabilities(self.audio, model, torch.device("cpu"), context_seconds=context)
        self.assertFalse(model.inputs)
        for blend in ("crop", "weighted", "equal"):
            with self.subTest(blend=blend):
                model = AudioMarkerModel()
                cancel = threading.Event()
                with self.assertRaises(InterruptedError):
                    predict_probabilities(self.audio, model, torch.device("cpu"),
                                          progress=lambda *_: cancel.set(), cancel=cancel, context_seconds=2,
                                          overlap_blend=blend)
                self.assertEqual(len(model.inputs), 1)
        with self.assertRaisesRegex(ValueError, "Overlap blend"):
            predict_probabilities(self.audio, model, torch.device("cpu"), overlap_blend="unknown")
        with patch("piano_ml.inference.load_model") as load:
            with self.assertRaisesRegex(ValueError, "Overlap blend"):
                transcribe_audio(self.audio, "fake.pt", overlap_blend="unknown")
            load.assert_not_called()

    def test_result_records_the_context_actually_used(self):
        for blend in ("crop", "weighted", "equal"):
            with self.subTest(blend=blend):
                model = AudioMarkerModel()
                with patch("piano_ml.inference.load_model", return_value=model), \
                        patch("piano_ml.inference.decode_outputs", return_value={"notes": [], "chords": [], "pedals": []}) as decoder:
                    result = transcribe_audio(self.audio, "fake.pt", device="cpu", context_seconds=1,
                                              overlap_blend=blend)
                self.assertEqual(result["inference_config"],
                                 {"chunk_seconds": 4, "context_seconds": 1, "overlap_seconds": 2,
                                  "overlap_blend": blend})
                decoder.assert_called_once()
                self.assertEqual(decoder.call_args.args[0]["frame"].shape, (402, 88))
                self.assertEqual(len(model.inputs), 3)

    def test_unequal_predictions_follow_the_specified_weights_for_all_heads(self):
        head_values = {"frame": (0.2, 0.8), "onset": (0.1, 0.9), "offset": (0.3, 0.7),
                       "velocity": (0.4, 0.6), "pedal": (0.25, 0.75)}

        class UnequalModel:
            architecture = "onsets"

            def __init__(self):
                self.calls = 0

            def __call__(self, samples):
                frames = samples.shape[1] // HOP_LENGTH + 1
                output = {}
                for head, values in head_values.items():
                    value = values[min(self.calls, 1)]
                    logits = torch.logit(torch.tensor(value))
                    output[head] = torch.full((1, frames, 1 if head == "pedal" else 88), logits.item())
                    output[head][:, -1] = 20  # the extra FFT endpoint must not contribute
                self.calls += 1
                return output

        model = UnequalModel()
        _, output = predict_probabilities(self.audio, model, torch.device("cpu"), context_seconds=1,
                                         overlap_blend="weighted")
        self.assertEqual(model.calls, 3)
        for head, (earlier, later) in head_values.items():
            for time, later_weight in ((2, 0), (3, 0), (3.5, 0.25), (4, 0.5),
                                       (4.5, 0.75), (5, 1), (6, 1)):
                with self.subTest(head=head, time=time):
                    expected = earlier * (1 - later_weight) + later * later_weight
                    np.testing.assert_allclose(output[head][round(time / 0.02)], expected, rtol=0, atol=1e-7)

    def test_equal_blend_is_a_probability_mean_throughout_each_overlap(self):
        values = {"frame": (0.1, 0.8, 0.2), "onset": (0.3, 0.7, 0.9), "offset": (0.2, 0.6, 0.1),
                  "velocity": (0.6, 0.9, 0.4), "pedal": (0.7, 0.4, 0.8)}

        class UnequalModel:
            def __init__(self):
                self.calls = 0

            def __call__(self, samples):
                frames = samples.shape[1] // HOP_LENGTH + 1
                output = {}
                for head, probabilities in values.items():
                    logits = torch.logit(torch.tensor(probabilities[self.calls])).item()
                    output[head] = torch.full((1, frames, 1 if head == "pedal" else 88), logits)
                    output[head][:, -1] = 20  # endpoint has no contribution
                self.calls += 1
                return output

        model = UnequalModel()
        _, output = predict_probabilities(self.audio, model, torch.device("cpu"),
                                         context_seconds=1, overlap_blend="equal")
        self.assertEqual(model.calls, 3)
        for head, (first, second, third) in values.items():
            self.assertEqual(output[head].shape, (402, 1 if head == "pedal" else 88))
            for time, expected in ((2, first), (2.98, first), (3, (first + second) / 2),
                                   (3.5, (first + second) / 2), (4, (first + second) / 2),
                                   (4.5, (first + second) / 2), (4.98, (first + second) / 2),
                                   (5, second), (6.98, second), (7, (second + third) / 2),
                                   (8.02, (second + third) / 2)):
                with self.subTest(head=head, time=time):
                    np.testing.assert_allclose(output[head][round(time / 0.02)], expected, rtol=0, atol=1e-7)

    def test_short_and_partial_recordings_have_complete_finite_coverage(self):
        for duration in (0.001, 0.1, 3.98, 4, 4.02, 5, 8, 8.001, 8.04, 12.2):
            samples = (np.arange(round(duration * SAMPLE_RATE)) // HOP_LENGTH).astype("<i2")
            with wave.open(str(self.audio), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(SAMPLE_RATE)
                wav.writeframes(samples.tobytes())
            expected = torch.from_numpy(samples[::HOP_LENGTH].astype(np.float32) / 32768).sigmoid().numpy()
            for overlap, blend in product((0, 1, 2, 3, 4), ("weighted", "equal")):
                with self.subTest(duration=duration, overlap=overlap, blend=blend):
                    model = AudioMarkerModel()
                    _, output = predict_probabilities(self.audio, model, torch.device("cpu"),
                                                     context_seconds=overlap / 2, overlap_blend=blend)
                    self.assertEqual(output["frame"].shape, (len(expected), 88))
                    np.testing.assert_allclose(output["frame"][:, 0], expected, rtol=0, atol=1e-7)
                    self.assertTrue(np.isfinite(output["pedal"]).all())

    def test_repeated_notes_and_pedals_survive_boundaries(self):
        flags = np.zeros(402, dtype="<i2")
        note_times = ((3.8, 4.2), (4.4, 4.8), (7.8, 8.02))
        pedal_times = ((3.7, 4.7), (7.7, 8.04))
        for start, end in note_times:
            first, last = round(start / 0.02), round(end / 0.02)
            flags[first:last] |= 1
            flags[first] |= 2
            flags[last] |= 4
        for start, end in pedal_times:
            flags[round(start / 0.02):round(end / 0.02)] |= 8
        with wave.open(str(self.audio), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(np.repeat(flags, HOP_LENGTH).tobytes())

        class EventModel:
            architecture = "onsets"

            def __call__(self, samples):
                marks = torch.round(samples[:, ::HOP_LENGTH] * 32768).long()
                marks = torch.nn.functional.pad(marks, (0, samples.shape[1] // HOP_LENGTH + 1 - marks.shape[1]))
                output = {head: torch.full((1, marks.shape[1], 88), -20.0)
                          for head in ("frame", "onset", "offset", "velocity")}
                for head, bit in (("frame", 1), ("onset", 2), ("offset", 4)):
                    output[head][:, :, 39] = torch.where(marks & bit != 0, 20.0, -20.0)
                output["velocity"][:, :, 39] = torch.logit(torch.tensor(100 / 127)).item()
                output["pedal"] = torch.where(marks & 8 != 0, 20.0, -20.0).unsqueeze(-1)
                return output

        for overlap, blend in product((1, 2, 3, 4), ("weighted", "equal")):
            with self.subTest(overlap=overlap, blend=blend):
                _, probabilities = predict_probabilities(self.audio, EventModel(), torch.device("cpu"),
                                                        context_seconds=overlap / 2, overlap_blend=blend)
                decoded = decode_outputs(probabilities, 8.04, threshold=0.5, min_note_seconds=0.04)
                self.assertEqual([(note["start"], note["end"]) for note in decoded["notes"]], list(note_times))
                self.assertTrue(all(note["velocity"] == 100 for note in decoded["notes"]))
                self.assertEqual([(pedal["start"], pedal["end"]) for pedal in decoded["pedals"]], list(pedal_times))

    def test_zero_overlap_is_exactly_the_existing_crop_behavior(self):
        _, crop = predict_probabilities(self.audio, AudioMarkerModel(), torch.device("cpu"), context_seconds=0)
        for blend in ("weighted", "equal"):
            _, blended = predict_probabilities(self.audio, AudioMarkerModel(), torch.device("cpu"),
                                              context_seconds=0, overlap_blend=blend)
            for head in crop:
                np.testing.assert_array_equal(crop[head], blended[head])

    def test_incomplete_weighted_model_output_is_rejected(self):
        class ShortOutput:
            def __call__(self, samples):
                return torch.zeros((1, 2, 88))
        for blend in ("weighted", "equal"):
            with self.assertRaisesRegex(ValueError, "does not cover"):
                predict_probabilities(self.audio, ShortOutput(), torch.device("cpu"),
                                      context_seconds=1, overlap_blend=blend)

    def test_v8_accepts_the_larger_contexts(self):
        model = FourierRecurrentPianoNet(feature_width=32, hidden_size=16, gru_layers=1,
                                        fourier_layers=1, fourier_modes=3, dropout=0,
                                        rnn_backend="native").eval()
        for context, blend in product((1, 2), ("weighted", "equal")):
            with self.subTest(context=context, blend=blend):
                _, output = predict_probabilities(self.audio, model, torch.device("cpu"), context_seconds=context,
                                                 overlap_blend=blend)
                self.assertEqual(set(output), {"frame", "onset", "offset", "velocity", "pedal"})
                for head, probabilities in output.items():
                    self.assertEqual(probabilities.shape, (402, 1 if head == "pedal" else 88))
                    self.assertTrue(np.isfinite(probabilities).all())


if __name__ == "__main__":
    unittest.main()
