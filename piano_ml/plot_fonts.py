"""Installed font fallbacks for Unicode filenames and chart labels."""

from functools import lru_cache

from matplotlib import font_manager


@lru_cache(maxsize=1)
def plot_font_families() -> tuple[str, ...]:
    """Keep the usual font, then fall back to installed CJK fonts per glyph."""
    installed = {font.name for font in font_manager.fontManager.ttflist}
    candidates = (
        "DejaVu Sans", "Microsoft JhengHei", "Microsoft YaHei", "Yu Gothic",
        "Meiryo", "Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans CJK TC",
        "Noto Sans SC", "Noto Sans TC", "WenQuanYi Zen Hei", "Arial Unicode MS",
    )
    return tuple(name for name in candidates if name in installed) or ("sans-serif",)
