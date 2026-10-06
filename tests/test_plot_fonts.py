import unittest
import warnings

from matplotlib import font_manager
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ft2font import FT2Font
from matplotlib.text import Text

from piano_ml.plot_fonts import plot_font_families
from piano_ml.score import draw_staff
from piano_ml.viewer import draw_prediction


class PlotFontsTest(unittest.TestCase):
    def test_unicode_filenames_and_labels_render_without_missing_glyphs(self):
        label = "钢琴「练习曲」"
        fonts = [FT2Font(font_manager.findfont(name, fallback_to_default=False))
                 for name in plot_font_families()]
        if not all(any(font.get_char_index(ord(char)) for font in fonts) for char in label):
            self.skipTest("Rendering check needs an installed CJK font.")
        result = {
            "audio": f"{label}.wav", "model": "model「v8」.pt", "duration": 4.0,
            "notes": [{"pitch": 60, "name": label, "start": 0.5, "end": 1.0}],
            "chords": [{"name": label, "start": 0.5, "end": 1.0}],
        }
        for view in ("timeline", "staff"):
            with self.subTest(view=view), warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always", UserWarning)
                figure = Figure(figsize=(11, 6))
                if view == "timeline":
                    draw_prediction(figure, result)
                else:
                    draw_staff(figure, result, names=True)
                FigureCanvasAgg(figure).draw()
                glyph_warnings = [str(item.message) for item in captured
                                  if "Glyph" in str(item.message)]
                self.assertEqual(glyph_warnings, [])
                texts = [artist.get_text() for artist in figure.findobj(Text)]
                self.assertIn(label, texts)
                if view == "timeline":
                    self.assertTrue(any(f"{label}.wav" in text for text in texts))


if __name__ == "__main__":
    unittest.main()
