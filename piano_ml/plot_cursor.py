"""Move plot cursors without redrawing static notes or notation."""

from matplotlib.transforms import Bbox


class PlotCursor:
    def __init__(self, canvas):
        self.canvas = canvas
        self.artists = []
        self.background = None
        self.bounds = None
        self.geometry = None
        self.axes_callbacks = []
        self.callbacks = [canvas.mpl_connect("draw_event", self._on_draw),
                          canvas.mpl_connect("resize_event", self.invalidate)]

    def set_artists(self, artists):
        for axis, callback in self.axes_callbacks:
            axis.callbacks.disconnect(callback)
        self.axes_callbacks.clear()
        self.artists = list(artists)
        for artist in self.artists:
            artist.set_animated(self.canvas.supports_blit)
        for axis in {artist.axes for artist in self.artists}:
            for event in ("xlim_changed", "ylim_changed"):
                self.axes_callbacks.append((axis, axis.callbacks.connect(event, self.invalidate)))
        self.invalidate()

    def invalidate(self, _event=None):
        self.background = self.bounds = None
        self.geometry = None

    def _cursor_bounds(self):
        boxes = []
        for artist in self.artists:
            if artist.get_visible():
                padding = max(3, artist.get_linewidth() * self.canvas.figure.dpi / 144 + 2)
                box = Bbox.intersection(artist.get_window_extent().padded(padding), artist.axes.bbox)
                if box is not None:
                    boxes.append(box)
        return Bbox.union(boxes) if boxes else None

    def _draw_artists(self):
        for artist in self.artists:
            if artist.get_visible():
                artist.axes.draw_artist(artist)

    def _on_draw(self, event):
        # Export may use a different canvas or DPI. Never cache that renderer.
        if event.canvas is not self.canvas or event.canvas.is_saving():
            self.invalidate()
            return
        if not self.artists or not self.canvas.supports_blit:
            return
        self.background = self.canvas.copy_from_bbox(self.canvas.figure.bbox)
        self.geometry = (*self.canvas.figure.bbox.bounds, self.canvas.figure.dpi)
        self._draw_artists()
        self.bounds = self._cursor_bounds()

    def update(self):
        if not self.artists:
            return
        if not self.canvas.supports_blit:
            self.canvas.draw_idle()
            return
        if self.background is None or self.geometry != (*self.canvas.figure.bbox.bounds, self.canvas.figure.dpi):
            self.invalidate()
            self.canvas.draw_idle()
            return
        previous = self.bounds
        self.canvas.restore_region(self.background)
        self._draw_artists()
        self.bounds = self._cursor_bounds()
        dirty = [box for box in (previous, self.bounds) if box is not None]
        if dirty:
            self.canvas.blit(Bbox.union(dirty))

    def close(self):
        self.set_artists([])
        for callback in self.callbacks:
            self.canvas.mpl_disconnect(callback)
