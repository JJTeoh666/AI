import io
import random
import unittest
from unittest.mock import patch

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from piano_ml.plot_cursor import PlotCursor
from piano_ml.score import draw_staff
from piano_ml.viewer import EventIndex, PredictionIndex, active_at
from tests import test_app_comparison as comparison


class EventIndexTest(unittest.TestCase):
    def test_matches_scan_for_unsorted_events_long_holds_and_arbitrary_seeks(self):
        randomizer = random.Random(42)
        notes = [{"name": f"note{index % 88}", "pitch": 21 + index % 88,
                  "start": randomizer.randrange(0, 1000) / 10,
                  "end": 0} for index in range(1500)]
        for note in notes:
            note["end"] = note["start"] + randomizer.randrange(1, 100) / 10
        notes.extend(({"name": "held", "pitch": 21, "start": 0, "end": 120},
                      {"name": "repeated", "pitch": 60, "start": 10, "end": 11},
                      {"name": "repeated", "pitch": 60, "start": 10, "end": 11}))
        result = {"notes": notes, "chords": notes[::20]}
        index = PredictionIndex(result)
        for seconds in (0, 10, 11, 120, -1, 25, 119, 50, 5, *[randomizer.random() * 120 for _ in range(80)]):
            self.assertEqual(index.active_at(seconds), active_at(result, seconds))
        for start, end in ((0, 4), (10, 14), (100, 120), (11, 11), (8, 3)):
            expected = [event for event in notes if event["start"] < end and event["end"] > start] if end > start else []
            self.assertEqual(index.notes.overlapping(start, end), expected)
            expected = [event for event in notes if start <= event["start"] < end]
            self.assertEqual(index.notes.starting_between(start, end), expected)
        self.assertEqual(EventIndex([]).at(0), [])

    def test_indexed_staff_keeps_notes_crossing_page_boundaries(self):
        result = {"duration": 12, "notes": [
            {"pitch": 24, "name": "C1", "start": 0, "end": 12},
            {"pitch": 60, "name": "C4", "start": 3.9, "end": 4.5},
            {"pitch": 65, "name": "F4", "start": 8, "end": 9}],
            "chords": [{"name": "C", "start": 4, "end": 5}]}
        images = []
        for index in (None, PredictionIndex(result)):
            figure = Figure(figsize=(8, 4))
            canvas = FigureCanvasAgg(figure)
            draw_staff(figure, result, page=1, names=True, event_index=index)
            canvas.draw()
            images.append(np.asarray(canvas.buffer_rgba()).copy())
        np.testing.assert_array_equal(*images)


class PlotCursorTest(unittest.TestCase):
    def setUp(self):
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvasAgg(self.figure)
        self.axis = self.figure.add_subplot()
        self.axis.plot([0, 1, 2, 3], [0, 1, 0, 1])
        self.line = self.axis.axvline(.1, color="red")
        self.cursor = PlotCursor(self.canvas)
        self.cursor.set_artists([self.line])
        self.canvas.draw()

    def tearDown(self):
        self.cursor.close()

    def assert_matches_full_render(self):
        actual = np.asarray(self.canvas.buffer_rgba()).copy()
        self.canvas.draw()
        np.testing.assert_array_equal(actual, np.asarray(self.canvas.buffer_rgba()))

    def test_cached_moves_do_not_redraw_and_leave_no_old_cursor_trails(self):
        for seconds in (.5, 1.8, .2, 3, -5):
            self.line.set_xdata([seconds, seconds])
            with patch.object(self.canvas, "draw") as draw, patch.object(self.canvas, "draw_idle") as idle:
                self.cursor.update()
            draw.assert_not_called()
            idle.assert_not_called()
            self.assert_matches_full_render()
        self.line.set_visible(False)
        self.cursor.update()
        self.assert_matches_full_render()

    def test_zoom_resize_and_export_refresh_the_static_background(self):
        self.axis.set_xlim(0, 1)
        self.assertIsNone(self.cursor.background)
        self.cursor.update()
        self.assert_matches_full_render()
        self.figure.set_size_inches(7, 3)
        self.line.set_xdata([.5, .5])
        self.cursor.update()
        self.assert_matches_full_render()
        self.figure.savefig(io.BytesIO(), format="png", dpi=150)
        self.assertIsNone(self.cursor.background)
        self.cursor.update()
        self.assert_matches_full_render()

    def test_fallback_canvas_uses_normal_drawing(self):
        self.canvas.supports_blit = False
        self.cursor.set_artists([self.line])
        self.assertFalse(self.line.get_animated())
        with patch.object(self.canvas, "draw_idle") as idle:
            self.cursor.update()
        idle.assert_called_once()


class PlaybackUITest(unittest.TestCase):
    setUp = comparison.AppComparisonTest.setUp
    tearDown = comparison.AppComparisonTest.tearDown
    result = comparison.AppComparisonTest.result

    def test_only_visible_chart_updates_without_full_redraws(self):
        self.app.show_result(self.result())
        with patch.object(self.app.canvas, "draw_idle") as hidden, \
                patch.object(self.app.staff_canvas, "draw_idle") as full, \
                patch.object(self.app.staff_canvas, "blit", wraps=self.app.staff_canvas.blit) as blit:
            for index in range(20):
                self.app.update_position(index * .1)
        hidden.assert_not_called()
        full.assert_not_called()
        self.assertEqual(blit.call_count, 20)
        # The Tk image contains the restored old stripe and the new cursor.
        line = self.app.staff_line
        x = round(line.get_window_extent().x0)
        y = round(self.app.staff_axis.bbox.y0 + self.app.staff_axis.bbox.height / 2)
        pixels = np.asarray(self.app.staff_canvas.buffer_rgba())
        tk_y = pixels.shape[0] - 1 - y
        self.assertEqual(self.app.staff_canvas._tkphoto.get(x, tk_y), tuple(pixels[tk_y, x, :3]))
        self.app.tabs.select(self.app.timeline_tab)
        self.root.update()
        with patch.object(self.app.staff_cursor, "update") as hidden, \
                patch.object(self.app.canvas, "draw_idle") as full:
            self.app.update_position(.5)
        hidden.assert_not_called()
        full.assert_not_called()

    def test_hidden_staff_catches_up_once_and_final_endpoint_does_not_redraw(self):
        self.app.show_result(self.result())
        self.app.tabs.select(self.app.tabs.tabs()[2])
        self.root.update()
        with patch.object(self.app, "redraw_staff", wraps=self.app.redraw_staff) as redraw:
            self.app.update_position(4.5)
            self.app.update_position(5)
            redraw.assert_not_called()
            self.assertEqual(self.app.staff_page, 1)
            self.app.tabs.select(self.app.staff_tab)
            self.root.update()
            self.assertEqual(redraw.call_count, 1)
            self.assertEqual(self.app.staff_window, (4, 8))
            redraw.reset_mock()
            for _ in range(5):
                self.app.update_position(8)
            redraw.assert_not_called()
            self.assertTrue(self.app.staff_line.get_visible())

    def test_seek_previews_are_coalesced_and_release_uses_final_position(self):
        self.app.show_result(self.result())
        self.app.begin_seek()
        with patch.object(self.app, "move_cursor", wraps=self.app.move_cursor) as move:
            for index in range(100):
                self.app.seek_value.set(index / 20)
                self.app.preview_seek(index / 20)
            move.assert_not_called()
            self.assertIsNotNone(self.app._seek_preview_after)
            self.app.end_seek()
            move.assert_called_once_with(4.95)
        self.assertIsNone(self.app._seek_preview_after)
        self.assertEqual(self.app.playback_position, 4.95)

    def test_poll_uses_fast_refresh_only_while_playing(self):
        self.app.show_result(self.result())
        self.app.player = comparison.FakePlayer()
        self.app.start_playback(self.audio, "test")
        with patch.object(self.root, "after", wraps=self.root.after) as after:
            self.app.poll()
        self.assertEqual(after.call_args.args[0], 33)
        self.app.pause()
        with patch.object(self.root, "after", wraps=self.root.after) as after:
            self.app.poll()
        self.assertEqual(after.call_args.args[0], 100)


if __name__ == "__main__":
    unittest.main()
