"""Tkinter desktop app for choosing models and exploring piano predictions."""

import json
import queue
import shutil
import threading
import time
import tempfile
import tkinter as tk
import wave
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from .inference import transcribe_audio, load_model
from .bytedance import transcribe_bytedance
import torch
from .thresholds import Thresholds, saved_thresholds
from .score import draw_staff
from .synthesis import render_result_wav, piano_sound_name
from .playback import WavPlayer, playback_available
from .plot_cursor import PlotCursor
from .viewer import PredictionIndex, draw_prediction, read_prediction, waveform_envelope

PROJECT = Path(__file__).resolve().parent.parent
OVERLAP_METHODS = {"Weighted average": "weighted", "50/50 average": "equal", "Keep center": "crop"}
RESULT_LABELS = {"Your model": "own", "ByteDance": "bytedance"}
PLAYBACK_SOURCES = (*RESULT_LABELS, "Original recording")
PLAYBACK_REFRESH_MS = 33


class PianoApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("Piano Notes Studio")
        root.geometry("1260x850")
        root.minsize(900, 650)
        root.configure(bg="#f4f6fa")
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TFrame", background="#f4f6fa")
        style.configure("TLabel", background="#f4f6fa", foreground="#344054", font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI", 19, "bold"), foreground="#15243b")
        style.configure("TButton", padding=(10, 6), font=("Segoe UI", 10))
        style.configure("Accent.TButton", background="#126b60", foreground="white")
        style.map("Accent.TButton", background=[("active", "#0e574d")])
        style.configure("Treeview", rowheight=26, font=("Segoe UI", 10))
        style.configure("Card.TFrame", background="white")
        style.configure("Card.TLabel", background="white", foreground="#344054")
        style.configure("CardTitle.TLabel", background="white", foreground="#15243b", font=("Segoe UI", 11, "bold"))
        style.configure("Muted.TLabel", foreground="#667085")
        style.configure("Card.TRadiobutton", background="white", font=("Segoe UI", 11, "bold"))
        style.configure("TNotebook", background="#f4f6fa", borderwidth=0)
        style.configure("TNotebook.Tab", padding=(16, 9), font=("Segoe UI", 10))
        style.map("TNotebook.Tab", background=[("selected", "white")], foreground=[("selected", "#126b60")])
        style.configure("Treeview.Heading", padding=(8, 7), font=("Segoe UI", 10, "bold"))
        self.model = tk.StringVar()
        self.audio = tk.StringVar()
        self.threshold = tk.DoubleVar(value=0.5)
        self.onset_threshold = tk.DoubleVar(value=0.5)
        self.offset_threshold = tk.DoubleVar(value=0.5)
        self.threshold_boxes = []
        self.threshold_details = tk.StringVar(value="")
        self.use_model_threshold = tk.BooleanVar(value=True)
        self.device = tk.StringVar(value="auto")
        self.overlap_seconds = tk.DoubleVar(value=2.0)
        self.overlap_method = tk.StringVar(value="Weighted average")
        self.status = tk.StringVar(value="Choose a piano WAV. Analyze your model or compare with ByteDance.")
        self.summary = tk.StringVar(value="No analysis loaded")
        self.cursor_text = tk.StringVar(value="Click the timeline to inspect notes at a moment.")
        self.model_paths: dict[str, Path] = {}
        self.result = None
        self.prediction_index = None
        self.results, self.result_envelopes, self.audio_cache = {}, {}, {}
        self.result_kind = "own"
        self.recording_path = None
        self.selected_result = tk.StringVar(value="Your model")
        self.playback_source = tk.StringVar(value="Your model")
        self.result_summaries = {key: tk.StringVar(value="No analysis yet") for key in ("own", "bytedance")}
        self.seek_value = tk.DoubleVar(value=0)
        self.position_text = tk.StringVar(value="0:00 / 0:00")
        self.playback_position = 0.0
        self.playback_path = None
        self.seeking = False
        self.seek_was_playing = False
        self._seek_preview_after = None
        self.axes, self.cursor_lines = [], []
        self.messages = queue.Queue()
        self.volume = tk.DoubleVar(value=50)
        self.volume_text = tk.StringVar(value="50%")
        self.gain = tk.DoubleVar(value=0)
        self.gain_text = tk.StringVar(value="+0.0 dB")
        self.player = WavPlayer(on_error=lambda error: self.messages.put(("playback_error", error))) if playback_available() else None
        self.cancel_event = threading.Event()
        self.busy = False
        self.playing_since = None
        self.playing_duration = 0
        self.staff_bpm = tk.DoubleVar(value=120)
        self.staff_zoom = tk.DoubleVar(value=125)
        self.staff_zoom_text = tk.StringVar(value="125%")
        self.staff_tempo = 120.0
        self.staff_names = tk.BooleanVar(value=False)
        self.staff_page_text = tk.StringVar(value="No score loaded")
        self.staff_page = 0
        self.staff_pages = 1
        self.staff_window = (0, 4)
        self.staff_axis = self.staff_line = None
        self._staff_dirty = False
        self.cursor_seconds = 0
        self.audio_cancel = threading.Event()
        self.audio_temp = None
        self.audio_thread = None
        self.rendered_result = None
        self.task_kind = None
        self._build()
        self.refresh_models()
        root.protocol("WM_DELETE_WINDOW", self.close)
        self._poll_after = root.after(100, self.poll)

    def _build(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        body = ttk.Frame(self.root)
        body.grid(row=0, column=0, sticky="nsew")
        self.scroll_canvas = tk.Canvas(body, background="#f4f6fa", highlightthickness=0,
                                       yscrollincrement=24)
        self.page_scrollbar = ttk.Scrollbar(body, orient="vertical", command=self.scroll_canvas.yview)
        self.page_scrollbar.pack(side="right", fill="y")
        self.scroll_canvas.configure(yscrollcommand=self.page_scrollbar.set)
        self.scroll_canvas.pack(side="left", fill="both", expand=True)
        outer = self.scroll_content = ttk.Frame(self.scroll_canvas, padding=18)
        self.scroll_window = self.scroll_canvas.create_window(0, 0, window=outer, anchor="nw")
        outer.bind("<Configure>", self.fit_scroll_content)
        self.scroll_canvas.bind("<Configure>", self.fit_scroll_content)
        self._wheel_remainder = 0.0
        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 16))
        ttk.Label(header, text="Piano Notes Studio", style="Title.TLabel").pack(side="left")
        ttk.Button(header, text="Open result JSON…", command=self.open_json).pack(side="right")
        ttk.Label(outer, text="Transcribe a recording. Compare two models. Listen to the difference.",
                  style="Muted.TLabel").pack(anchor="w", pady=(0, 16))
        inputs = ttk.Frame(outer)
        inputs.pack(fill="x")
        inputs.columnconfigure(0, weight=1, uniform="inputs")
        inputs.columnconfigure(1, weight=1, uniform="inputs")
        controls = ttk.Frame(inputs, style="Card.TFrame", padding=14)
        controls.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        controls.columnconfigure(1, weight=1)
        ttk.Label(controls, text="01   YOUR MODEL", style="CardTitle.TLabel").grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))
        self.models_box = ttk.Combobox(controls, textvariable=self.model, state="readonly")
        self.models_box.grid(row=1, column=0, columnspan=2, sticky="ew", pady=4)
        self.models_box.bind("<<ComboboxSelected>>", lambda _event: self.preview_thresholds())
        self.model_browse_button = ttk.Button(controls, text="Browse…", command=self.browse_model)
        self.model_browse_button.grid(row=1, column=2, padx=(8, 0))
        self.refresh_button = ttk.Button(controls, text="Refresh models", command=self.refresh_models)
        self.refresh_button.grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Label(controls, text="PyTorch checkpoint · V8.1–V12.1", style="Card.TLabel").grid(row=2, column=1, columnspan=2, sticky="e")
        audio_card = ttk.Frame(inputs, style="Card.TFrame", padding=14)
        audio_card.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        audio_card.columnconfigure(0, weight=1)
        ttk.Label(audio_card, text="02   PIANO RECORDING", style="CardTitle.TLabel").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))
        self.audio_entry = ttk.Entry(audio_card, textvariable=self.audio, state="readonly")
        self.audio_entry.grid(row=1, column=0, sticky="ew", pady=4)
        self.audio_browse_button = ttk.Button(audio_card, text="Choose WAV…", command=self.browse_audio)
        self.audio_browse_button.grid(row=1, column=1, padx=(8, 0))
        ttk.Label(audio_card, text="16-bit PCM WAV · Both models use the same recording",
                  style="Card.TLabel").grid(row=2, column=0, columnspan=2, sticky="w", pady=(10, 0))
        self.advanced_button = ttk.Button(outer, text="▸ Analysis settings", command=self.toggle_settings)
        self.advanced_button.pack(anchor="w", pady=(12, 0))
        self.advanced = ttk.Frame(outer, padding=(0, 6))
        threshold_options = ttk.Frame(self.advanced)
        threshold_options.pack(fill="x", pady=(4, 0))
        ttk.Checkbutton(threshold_options, text="Use model thresholds", variable=self.use_model_threshold,
                        command=self.update_threshold_controls).pack(side="left", padx=(0, 12))
        for label, variable in (("Frame", self.threshold), ("Onset", self.onset_threshold), ("Offset", self.offset_threshold)):
            ttk.Label(threshold_options, text=label).pack(side="left")
            box = ttk.Spinbox(threshold_options, from_=0.05, to=0.95, increment=0.05,
                              textvariable=variable, width=5, state="disabled")
            box.pack(side="left", padx=(6, 12))
            self.threshold_boxes.append(box)
        self.threshold_box = self.threshold_boxes[0]
        ttk.Label(self.advanced, textvariable=self.threshold_details, wraplength=820).pack(anchor="w", pady=(3, 0))
        overlap_options = ttk.Frame(self.advanced)
        overlap_options.pack(fill="x", pady=(8, 0))
        ttk.Label(overlap_options, text="Analysis overlap (s)").pack(side="left")
        self.overlap_box = ttk.Combobox(overlap_options, values=(0, 1, 2, 3, 4),
                                        textvariable=self.overlap_seconds, state="readonly", width=5)
        self.overlap_box.pack(side="left", padx=8)
        ttk.Label(overlap_options, text="Shared audio between 4-second sections; more overlap takes longer.",
                  wraplength=600).pack(side="left")
        blend_options = ttk.Frame(self.advanced)
        blend_options.pack(fill="x", pady=(6, 0))
        ttk.Label(blend_options, text="Overlap method").pack(side="left")
        self.overlap_method_box = ttk.Combobox(blend_options, values=tuple(OVERLAP_METHODS),
                                               textvariable=self.overlap_method, state="readonly", width=18)
        self.overlap_method_box.pack(side="left", padx=8)
        ttk.Label(blend_options, text="Weighted blends gradually; 50/50 gives both calculations equal weight.",
                  wraplength=550).pack(side="left")
        options = self.analysis_actions = ttk.Frame(outer)
        options.pack(fill="x", pady=(10, 10))
        ttk.Label(options, text="Device").pack(side="left")
        ttk.Combobox(options, values=("auto", "cpu", "cuda"), textvariable=self.device,
                     state="readonly", width=7).pack(side="left", padx=8)
        self.analyze_button = ttk.Button(options, text="Analyze", style="Accent.TButton", command=self.analyze)
        self.analyze_button.pack(side="left", padx=10)
        self.compare_button = ttk.Button(options, text="Compare with ByteDance", command=self.analyze_bytedance)
        self.compare_button.pack(side="left", padx=(0, 8))
        self.cancel_button = ttk.Button(options, text="Cancel", command=self.cancel_event.set, state="disabled")
        self.cancel_button.pack(side="left")
        ttk.Button(options, text="Export JSON…", command=self.export_json).pack(side="right")
        ttk.Button(options, text="Save view…", command=self.export_plot).pack(side="right", padx=8)
        comparison = ttk.Frame(outer)
        comparison.pack(fill="x", pady=(4, 12))
        self.result_buttons = {}
        for column, (label, key) in enumerate(RESULT_LABELS.items()):
            comparison.columnconfigure(column, weight=1, uniform="results")
            card = ttk.Frame(comparison, style="Card.TFrame", padding=14)
            card.grid(row=0, column=column, sticky="nsew", padx=(0, 6) if column == 0 else (6, 0))
            button = ttk.Radiobutton(card, text=label, style="Card.TRadiobutton", variable=self.selected_result,
                                     value=label, command=self.select_result, state="disabled")
            button.pack(anchor="w")
            self.result_buttons[key] = button
            ttk.Label(card, textvariable=self.result_summaries[key], style="Card.TLabel").pack(anchor="w", pady=(6, 0))
            ttk.Label(card, text="Learned thresholds, velocity & sustain" if key == "own" else "Pretrained · Notes, velocity & sustain",
                      style="Card.TLabel").pack(anchor="w", pady=(3, 0))
        ttk.Label(outer, textvariable=self.summary, wraplength=820, font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(3, 8))
        tabs = self.tabs = ttk.Notebook(outer)
        staff_tab = self.staff_tab = ttk.Frame(tabs)
        timeline, notes_tab, chords_tab = (ttk.Frame(tabs) for _ in range(3))
        self.timeline_tab = timeline
        tabs.add(staff_tab, text="  Staff  ")
        tabs.add(timeline, text="  Timeline  ")
        tabs.add(notes_tab, text="  Notes  ")
        tabs.add(chords_tab, text="  Chords  ")
        staff_controls = ttk.Frame(staff_tab, padding=(6, 6))
        staff_controls.pack(fill="x")
        ttk.Label(staff_controls, text="Notation tempo (BPM)").pack(side="left")
        tempo = ttk.Spinbox(staff_controls, from_=30, to=300, increment=5, width=6,
                            textvariable=self.staff_bpm, command=self.redraw_staff)
        tempo.pack(side="left", padx=8)
        tempo.bind("<Return>", lambda _event: self.redraw_staff())
        ttk.Button(staff_controls, text="Update staff", command=self.redraw_staff).pack(side="left")
        ttk.Checkbutton(staff_controls, text="Note names", variable=self.staff_names,
                        command=self.redraw_staff).pack(side="left", padx=10)
        ttk.Button(staff_controls, text="Next ›", command=lambda: self.change_staff_page(1)).pack(side="right")
        ttk.Label(staff_controls, textvariable=self.staff_page_text).pack(side="right", padx=10)
        ttk.Button(staff_controls, text="‹ Previous", command=lambda: self.change_staff_page(-1)).pack(side="right")
        staff_zoom_row = ttk.Frame(staff_tab, padding=(6, 0, 6, 6))
        staff_zoom_row.pack(fill="x")
        ttk.Label(staff_zoom_row, text="Staff zoom").pack(side="left")
        self.staff_zoom_slider = ttk.Scale(staff_zoom_row, from_=75, to=200, orient="horizontal",
                                           length=180, variable=self.staff_zoom, command=self.update_staff_zoom)
        self.staff_zoom_slider.pack(side="left", padx=8)
        ttk.Label(staff_zoom_row, textvariable=self.staff_zoom_text, width=5).pack(side="left")
        self.staff_figure = Figure(figsize=(11, 6), dpi=125)
        self.staff_figure.text(0.5, 0.5, "Load audio and analyze, or open a saved result",
                               ha="center", va="center", color="#667085", fontsize=14)
        self.staff_canvas = FigureCanvasTkAgg(self.staff_figure, master=staff_tab)
        self.staff_cursor = PlotCursor(self.staff_canvas)
        self.staff_toolbar = NavigationToolbar2Tk(self.staff_canvas, staff_tab, pack_toolbar=False)
        self.staff_toolbar.pack(side="bottom", fill="x")
        self.staff_canvas.get_tk_widget().configure(height=750)
        self.staff_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.staff_canvas.mpl_connect("button_press_event", self.on_staff_click)
        self.figure = Figure(figsize=(11, 6), dpi=100)
        self.figure.text(0.5, 0.5, "Load audio and analyze, or open a saved result",
                         ha="center", va="center", color="#667085", fontsize=14)
        self.canvas = FigureCanvasTkAgg(self.figure, master=timeline)
        self.timeline_cursor = PlotCursor(self.canvas)
        toolbar_row = ttk.Frame(timeline)
        toolbar_row.pack(side="bottom", fill="x")
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_row, pack_toolbar=False)
        self.toolbar.pack(side="left")
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("button_press_event", self.on_plot_click)
        self.tabs.bind("<<NotebookTabChanged>>", self.on_view_change)
        self.notes_table = self._table(notes_tab, ("Note", "MIDI", "Start (s)", "End (s)", "Duration (s)", "Confidence"))
        self.chords_table = self._table(chords_tab, ("Chord", "Start (s)", "End (s)", "Duration (s)"))
        self.notes_table.bind("<<TreeviewSelect>>", self.on_note_select)
        self.chords_table.bind("<<TreeviewSelect>>", self.on_chord_select)
        playback = self.playback_controls = ttk.Frame(self.root, padding=(18, 10, 18, 0))
        playback.grid(row=1, column=0, sticky="ew")
        playback.columnconfigure(0, weight=1)
        transport = ttk.Frame(playback)
        transport.grid(row=0, column=0, sticky="ew")
        ttk.Label(transport, text="LISTEN", font=("Segoe UI", 10, "bold")).pack(side="left", padx=(0, 10))
        self.source_box = ttk.Combobox(transport, values=PLAYBACK_SOURCES, textvariable=self.playback_source,
                                       state="readonly", width=18)
        self.source_box.pack(side="left", padx=(0, 8))
        self.source_box.bind("<<ComboboxSelected>>", self.change_source)
        self.play_result_button = ttk.Button(transport, text="Play", style="Accent.TButton",
                                            command=self.toggle_play, state="normal" if self.player else "disabled")
        self.play_result_button.pack(side="left")
        ttk.Button(transport, text="Stop", command=self.stop).pack(side="left", padx=8)
        self.play_original_button = ttk.Button(transport, text="Original", command=self.play,
                                               state="normal" if self.player else "disabled")
        self.play_original_button.pack(side="left")
        self.export_audio_button = ttk.Button(transport, text="Export result WAV…", command=self.export_audio)
        self.export_audio_button.pack(side="right")
        seek_row = ttk.Frame(playback)
        seek_row.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        seek_row.columnconfigure(0, weight=1)
        self.seek_slider = ttk.Scale(seek_row, from_=0, to=1, orient="horizontal", variable=self.seek_value,
                                     command=self.preview_seek, state="disabled")
        self.seek_slider.grid(row=0, column=0, sticky="ew")
        self.seek_slider.bind("<ButtonPress-1>", self.begin_seek)
        self.seek_slider.bind("<ButtonRelease-1>", self.end_seek)
        self.seek_slider.bind("<KeyRelease>", self.keyboard_seek)
        ttk.Label(seek_row, textvariable=self.position_text, width=16, anchor="e").grid(row=0, column=1, padx=(8, 0))
        levels = ttk.Frame(playback)
        levels.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        levels.columnconfigure(1, weight=1)
        levels.columnconfigure(4, weight=1)
        ttk.Label(levels, text="Volume").grid(row=0, column=0, padx=(0, 8))
        self.volume_slider = ttk.Scale(levels, from_=0, to=100, orient="horizontal", length=160,
                                       variable=self.volume, command=self.update_volume,
                                       state="normal" if self.player else "disabled")
        self.volume_slider.grid(row=0, column=1, sticky="ew")
        ttk.Label(levels, textvariable=self.volume_text, width=5, anchor="e").grid(row=0, column=2, padx=(6, 18))
        ttk.Label(levels, text="Gain").grid(row=0, column=3, padx=(0, 8))
        self.gain_slider = ttk.Scale(levels, from_=0, to=24, orient="horizontal", length=160,
                                     variable=self.gain, command=self.update_gain,
                                     state="normal" if self.player else "disabled")
        self.gain_slider.grid(row=0, column=4, sticky="ew")
        ttk.Label(levels, textvariable=self.gain_text, width=9, anchor="e").grid(row=0, column=5, padx=(8, 18))
        ttk.Label(levels, text=piano_sound_name(), style="Muted.TLabel").grid(row=0, column=6)
        footer = ttk.Frame(self.root, padding=(18, 8, 18, 12))
        footer.grid(row=2, column=0, sticky="ew")
        ttk.Label(footer, textvariable=self.cursor_text, wraplength=820).pack(anchor="w", pady=(0, 4))
        self.progress = ttk.Progressbar(footer, mode="determinate")
        self.progress.pack(fill="x", pady=4)
        ttk.Label(footer, textvariable=self.status, wraplength=820).pack(anchor="w")
        tabs.pack(fill="both", expand=True, pady=(8, 0))
        self.root.bind("<MouseWheel>", self.scroll_page, add="+")
        self.root.bind("<Button-4>", self.scroll_page, add="+")
        self.root.bind("<Button-5>", self.scroll_page, add="+")

    def toggle_settings(self):
        if self.advanced.winfo_manager():
            self.advanced.pack_forget()
            self.advanced_button.configure(text="▸ Analysis settings")
        else:
            self.advanced.pack(fill="x", before=self.analysis_actions)
            self.advanced_button.configure(text="▾ Analysis settings")

    def fit_scroll_content(self, _event=None) -> None:
        width = self.scroll_canvas.winfo_width()
        # Leave height automatic so notebook/zoom requests resize the content.
        self.scroll_canvas.itemconfigure(self.scroll_window, width=width)
        self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox(self.scroll_window))

    def scroll_page(self, event):
        # Tables keep their own scrolling; playback sliders sit outside this viewport.
        if isinstance(event.widget, (ttk.Treeview, ttk.Combobox, ttk.Spinbox, ttk.Scale)):
            return
        canvas = self.scroll_canvas
        x, y = event.x_root - canvas.winfo_rootx(), event.y_root - canvas.winfo_rooty()
        if not (0 <= x < canvas.winfo_width() and 0 <= y < canvas.winfo_height()):
            return
        number = getattr(event, "num", None)
        if number in (4, 5):
            units = -3 if number == 4 else 3
        else:
            self._wheel_remainder += event.delta / 120
            steps = int(self._wheel_remainder)
            self._wheel_remainder -= steps
            units = -3 * steps
        if units:
            canvas.yview_scroll(units, "units")
            return "break"

    def update_staff_zoom(self, value) -> None:
        zoom = max(75, min(200, round(float(value))))
        self.staff_zoom_text.set(f"{zoom}%")
        self.staff_figure.set_dpi(zoom)
        self.staff_canvas.get_tk_widget().configure(height=round(600 * zoom / 100))
        self.staff_cursor.invalidate()
        self.staff_canvas.draw_idle()

    def _table(self, parent, columns):
        table = ttk.Treeview(parent, columns=columns, show="headings", selectmode="browse")
        scrollbar = ttk.Scrollbar(parent, command=table.yview)
        table.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        table.pack(fill="both", expand=True)
        for column in columns:
            table.heading(column, text=column)
            table.column(column, width=140, anchor="center")
        return table

    def refresh_models(self) -> None:
        for path in sorted((PROJECT / "checkpoints").glob("*.pt")):
            self.model_paths[path.name] = path
        self.models_box["values"] = list(self.model_paths)
        if not self.model.get() and self.model_paths:
            preferred = ("piano-v12.1.pt", "piano-v12.pt", "piano-v11.1.pt", "piano-v11.pt", "piano-v10.1.pt", "piano-v10.pt", "piano-v9.pt", "piano-v8.1.pt", "piano-v8.pt", "piano-v7.pt", "piano-v6.pt", "piano-v5.pt", "piano-v2-separate.pt", "piano-v2-tuned.pt", "piano-v2.pt", "piano-tuned.pt", "piano.pt", "piano-demo.pt")
            first = next((name for name in preferred if name in self.model_paths), next(iter(self.model_paths)))
            self.model.set(first)
        self.preview_thresholds()

    def preview_thresholds(self):
        if self.busy or self.model.get() not in self.model_paths:
            return
        try:
            model = load_model(self.model_paths[self.model.get()], torch.device("cpu"))
            self.display_thresholds(model.decoding_thresholds)
            if hasattr(model, "model_version"):
                self.threshold_details.set(f"V{model.model_version} — " + self.threshold_details.get())
        except Exception as error:
            self.threshold_details.set(f"Cannot read model thresholds: {error}")

    def display_thresholds(self, thresholds):
        summary = thresholds.summary()
        self.threshold.set(summary["frame"])
        self.onset_threshold.set(summary["onset"])
        self.offset_threshold.set(summary["offset"])
        if thresholds.pitch_dependent:
            text = "   |   ".join(f"{row['name'].capitalize()} {row['frame']:.2f}/{row['onset']:.2f}/{row['offset']:.2f}"
                                  for row in thresholds.registers())
            self.threshold_details.set("Model frame/onset/offset: " + text + ". Controls above are global overrides.")
        else:
            self.threshold_details.set("Model uses one frame/onset/offset threshold set for all pitches.")

    def browse_model(self) -> None:
        if self.busy:
            return
        selected = filedialog.askopenfilename(parent=self.root, title="Choose a PianoNet checkpoint",
                                              initialdir=PROJECT / "checkpoints",
                                              filetypes=[("PyTorch checkpoint", "*.pt *.pth"), ("All files", "*.*")])
        if selected:
            path = Path(selected).resolve()
            label = str(path)
            self.model_paths[label] = path
            self.models_box["values"] = list(self.model_paths)
            self.model.set(label)
            self.preview_thresholds()

    def browse_audio(self) -> None:
        if self.busy:
            return
        selected = filedialog.askopenfilename(parent=self.root, title="Choose piano audio",
                                              initialdir=PROJECT / "data",
                                              filetypes=[("16-bit PCM WAV", "*.wav")])
        if selected:
            try:
                with wave.open(selected, "rb") as audio:
                    if audio.getsampwidth() != 2 or audio.getcomptype() != "NONE" or not audio.getnframes():
                        raise ValueError("Choose a nonempty, uncompressed 16-bit PCM WAV.")
                    duration = audio.getnframes() / audio.getframerate()
            except (OSError, ValueError, wave.Error) as error:
                messagebox.showerror("Cannot open audio", str(error), parent=self.root)
                return
            self.clear_results()
            self.audio.set(selected)
            self.recording_path = str(Path(selected).resolve())
            self.configure_duration(duration)
            self.update_result_controls()
            self.status.set("Recording ready. Analyze your model or compare with ByteDance.")

    def clear_results(self):
        self.stop()
        self.results.clear()
        self.result_envelopes.clear()
        self.audio_cache.clear()
        self.result = self.rendered_result = None
        self.prediction_index = None
        self.recording_path = None
        self.axes, self.cursor_lines = [], []
        self.staff_axis = self.staff_line = None
        self._staff_dirty = False
        self.timeline_cursor.set_artists([])
        self.staff_cursor.set_artists([])
        self.staff_page = 0
        self.staff_pages = 1
        self.staff_page_text.set("No score loaded")
        self.summary.set("No analysis loaded")
        self.cursor_text.set("Analyze the recording to display notes and chords.")
        for variable in self.result_summaries.values():
            variable.set("No analysis yet")
        for table in (self.notes_table, self.chords_table):
            children = table.get_children()
            if children:
                table.delete(*children)
        for figure, canvas in ((self.figure, self.canvas), (self.staff_figure, self.staff_canvas)):
            figure.clear()
            figure.text(.5, .5, "Analyze a recording to see its notes", ha="center", va="center", color="#667085")
            canvas.draw_idle()
        self.configure_duration(0)
        self.update_result_controls()

    def ensure_recording(self, audio):
        identity = str(Path(audio).resolve())
        if self.recording_path != identity:
            self.clear_results()
            self.recording_path = identity

    def update_threshold_controls(self) -> None:
        for box in self.threshold_boxes:
            box.configure(state="disabled" if self.use_model_threshold.get() else "normal")
        if self.use_model_threshold.get():
            self.preview_thresholds()

    def analyze(self) -> None:
        if self.busy:
            return
        try:
            model = self.model_paths[self.model.get()]
            audio = Path(self.audio.get())
            thresholds = {} if self.use_model_threshold.get() else Thresholds(
                frame=float(self.threshold.get()), onset=float(self.onset_threshold.get()),
                offset=float(self.offset_threshold.get())).to_dict()
            overlap = float(self.overlap_seconds.get())
            if overlap not in (0, 1, 2, 3, 4):
                raise ValueError("Choose an analysis overlap from 0 to 4 seconds.")
            if self.overlap_method.get() not in OVERLAP_METHODS:
                raise ValueError("Choose a valid overlap method.")
            overlap_blend = OVERLAP_METHODS[self.overlap_method.get()]
            if not audio.is_file():
                raise ValueError("Choose an existing piano WAV file.")
            if not model.is_file():
                raise ValueError("Choose an existing model checkpoint.")
        except (KeyError, ValueError, tk.TclError) as error:
            messagebox.showerror("Choose inputs", str(error) if not isinstance(error, KeyError) else "Choose a model checkpoint.", parent=self.root)
            return
        self.ensure_recording(audio)
        self.stop()
        self.set_busy(True, "analysis")
        self.cancel_event.clear()
        self.progress["value"] = 0
        self.status.set(f"Loading {model.name} and analyzing {audio.name}…")
        device = self.device.get()
        threading.Thread(target=self._worker, args=(audio, model, thresholds, device, overlap, overlap_blend), daemon=True).start()

    def analyze_bytedance(self):
        if self.busy:
            return
        audio = Path(self.audio.get())
        if not audio.is_file():
            messagebox.showerror("Choose inputs", "Choose an existing piano WAV file.", parent=self.root)
            return
        self.ensure_recording(audio)
        self.stop()
        self.cancel_event.clear()
        self.set_busy(True, "bytedance")
        self.progress["value"] = 0
        self.status.set(f"Analyzing {audio.name} with ByteDance ({self.device.get()})…")
        threading.Thread(target=self._bytedance_worker, args=(audio, self.device.get()), daemon=True).start()

    def _bytedance_worker(self, audio, device="auto"):
        try:
            result = transcribe_bytedance(audio, device=device, cancel=self.cancel_event,
                                          progress=lambda done, total: self.messages.put(("progress", (done, total))))
            envelope = waveform_envelope(audio)
            if self.cancel_event.is_set():
                raise InterruptedError()
            self.messages.put(("result", (result, envelope)))
        except InterruptedError:
            self.messages.put(("cancelled", None))
        except Exception as error:
            self.messages.put(("error", str(error)))

    def set_busy(self, busy: bool, kind=None) -> None:
        self.busy = busy
        self.task_kind = kind if busy else None
        self.analyze_button.configure(state="disabled" if busy else "normal")
        self.compare_button.configure(state="disabled" if busy else "normal")
        for button in (self.model_browse_button, self.refresh_button, self.audio_browse_button):
            button.configure(state="disabled" if busy else "normal")
        self.models_box.configure(state="disabled" if busy else "readonly")
        self.overlap_box.configure(state="disabled" if busy else "readonly")
        self.overlap_method_box.configure(state="disabled" if busy else "readonly")
        self.cancel_button.configure(state="normal" if busy else "disabled", command=self.cancel_active)
        state = "normal" if self.player and not busy else "disabled"
        self.play_result_button.configure(state=state)
        self.play_original_button.configure(state=state)
        self.export_audio_button.configure(state="disabled" if busy else "normal")
        self.source_box.configure(state="disabled" if busy else "readonly")
        self.seek_slider.configure(state="disabled" if busy or not self.playing_duration else "normal")
        self.update_result_controls()

    def update_result_controls(self):
        for key, button in self.result_buttons.items():
            button.configure(state="normal" if key in self.results and not self.busy else "disabled")
        sources = [label for label, key in RESULT_LABELS.items() if key in self.results]
        if Path(self.audio.get()).is_file():
            sources.append("Original recording")
        self.source_box["values"] = sources
        if self.playback_source.get() not in sources and sources:
            self.playback_source.set(sources[0])

    def cancel_active(self) -> None:
        self.cancel_event.set()
        self.audio_cancel.set()
        self.status.set("Cancelling the current task…")

    def _worker(self, audio, model, thresholds, device, overlap_seconds, overlap_blend) -> None:
        try:
            started = time.monotonic()
            result = transcribe_audio(audio, model, device=device,
                                     **{f"{head}_threshold": value for head, value in thresholds.items()},
                                     progress=lambda done, total: self.messages.put(("progress", (done, total))),
                                     cancel=self.cancel_event, context_seconds=overlap_seconds / 2,
                                     overlap_blend=overlap_blend)
            result["analysis_seconds"] = time.monotonic() - started
            envelope = waveform_envelope(audio)
            if self.cancel_event.is_set():
                raise InterruptedError("Analysis cancelled.")
            self.messages.put(("result", (result, envelope)))
        except InterruptedError:
            self.messages.put(("cancelled", None))
        except Exception as error:
            self.messages.put(("error", str(error)))

    def poll(self) -> None:
        self.root.after_cancel(self._poll_after)
        try:
            while True:
                kind, value = self.messages.get_nowait()
                if kind in ("progress", "audio_progress"):
                    done, total = value
                    self.progress["value"] = 100 * done / total
                    self.status.set(f"{'Rendering detected notes' if kind == 'audio_progress' else 'Analyzing audio'}: {done}/{total} segments")
                elif kind == "playback_error":
                    self.stop()
                    self.status.set("Playback failed.")
                    messagebox.showerror("Cannot play audio", value, parent=self.root)
                else:
                    self.set_busy(False)
                    if kind == "result":
                        if self.cancel_event.is_set():
                            self.status.set("Analysis cancelled.")
                            continue
                        self.show_result(*value)
                        self.status.set("Analysis complete. Select a result card to view it, then Play to listen. Compare using the same recording.")
                    elif kind == "audio_ready":
                        path, export, result_kind = value
                        if self.audio_cancel.is_set():
                            self.status.set("Audio preparation cancelled.")
                            continue
                        self.audio_cache[result_kind] = path
                        if result_kind == self.result_kind:
                            self.rendered_result = path
                        if export:
                            self.status.set(f"Saved synthesized result audio: {export}")
                        else:
                            self.start_playback(path, self.playback_source.get())
                    elif kind == "cancelled":
                        self.status.set("Task cancelled.")
                    else:
                        self.status.set("Task failed.")
                        messagebox.showerror("Task failed", value, parent=self.root)
        except queue.Empty:
            pass
        if self.playing_since is not None:
            elapsed = self.current_position()
            if self.player is None or not self.player.is_playing:
                self._halt_playback()
                self.playback_position = self.playing_duration
                self.update_position(self.playback_position)
                self.status.set("Playback finished.")
            else:
                self.update_position(elapsed)
        interval = PLAYBACK_REFRESH_MS if self.playing_since is not None else 100
        self._poll_after = self.root.after(interval, self.poll)

    def show_result(self, result: dict, envelope=None, *, store=True, restore_settings=True, position=0.0) -> None:
        if result.get("engine") not in (None, "piano", "bytedance"):
            raise ValueError("This saved comparison engine is no longer supported. Analyze the recording with a current model.")
        self._halt_playback()
        result_kind = "bytedance" if result.get("engine") == "bytedance" else "own"
        if store:
            identity = result.get("audio")
            if identity:
                self.ensure_recording(identity)
            self.results[result_kind] = result
            self.result_envelopes[result_kind] = envelope
            self.audio_cache.pop(result_kind, None)
            seconds = result.get("analysis_seconds")
            timing = f" · {seconds:.2f} s analysis" if seconds is not None else ""
            self.result_summaries[result_kind].set(f"{len(result['notes'])} notes · {len(result['chords'])} chords{timing}")
        self.result_kind = result_kind
        label = "ByteDance" if result_kind == "bytedance" else "Your model"
        self.selected_result.set(label)
        self.playback_source.set(label)
        self.result = result
        self.prediction_index = PredictionIndex(result)
        self.rendered_result = self.audio_cache.get(result_kind)
        self.cursor_seconds = position
        self.staff_page = 0
        if restore_settings and result_kind == "own":
            overlap = result.get("inference_config", {}).get("overlap_seconds")
            if overlap in (0, 1, 2, 3, 4):
                self.overlap_seconds.set(overlap)
            blend = result.get("inference_config", {}).get("overlap_blend", "crop")
            self.overlap_method.set(next((label for label, mode in OVERLAP_METHODS.items() if mode == blend), "Keep center"))
        if restore_settings and result_kind == "own" and result.get("model") and Path(result["model"]).is_file():
            model_path = Path(result["model"]).resolve()
            label = next((name for name, path in self.model_paths.items() if path.resolve() == model_path), str(model_path))
            self.model_paths[label] = model_path
            self.models_box["values"] = list(self.model_paths)
            self.model.set(label)
        thresholds = saved_thresholds(result)
        if restore_settings and result_kind == "own":
            self.display_thresholds(thresholds)
        self.axes = draw_prediction(self.figure, result, envelope)
        self.cursor_lines = [axis.axvline(0, color="#e15759", linewidth=1, alpha=0.8) for axis in self.axes]
        self.timeline_cursor.set_artists(self.cursor_lines)
        if self.tabs.select() == str(self.timeline_tab):
            self.canvas.draw()
        self.toolbar.update()
        for table in (self.notes_table, self.chords_table):
            children = table.get_children()
            if children:
                table.delete(*children)
        for index, note in enumerate(result["notes"]):
            self.notes_table.insert("", "end", iid=str(index), values=(note["name"], note["pitch"],
                f"{note['start']:.3f}", f"{note['end']:.3f}", f"{note['end'] - note['start']:.3f}",
                f"{note.get('confidence', 1):.1%}"))
        for index, chord in enumerate(result["chords"]):
            self.chords_table.insert("", "end", iid=str(index), values=(chord["name"],
                f"{chord['start']:.3f}", f"{chord['end']:.3f}", f"{chord['end'] - chord['start']:.3f}"))
        model_name = Path(result["model"]).name if result.get("model") else "Saved result"
        summary = thresholds.summary()
        if result_kind == "bytedance":
            threshold_label = f"ByteDance: frame/onset/offset 0.10/0.30/0.30 · {len(result.get('pedals', []))} pedal intervals"
        elif thresholds.pitch_dependent:
            threshold_label = "pitch-dependent thresholds"
        else:
            threshold_label = f"frame/onset/offset {summary['frame']:.2f}/{summary['onset']:.2f}/{summary['offset']:.2f}"
        self.summary.set(f"{result['duration']:.2f} seconds   ·   {len(result['notes'])} note events   ·   "
                         f"{len(result['chords'])} chord spans   ·   {model_name}   ·   "
                         + threshold_label)
        self.tabs.select(self.staff_tab)
        self.staff_page = min(max(0, int(position / (8 * 60 / self.staff_tempo))),
                              max(0, int((result["duration"] - 1e-9) / (8 * 60 / self.staff_tempo))))
        self.redraw_staff(validate_tempo=False)
        self.configure_duration(result["duration"])
        self.playback_position = position
        self.update_position(position)
        self.update_result_controls()

    def select_result(self):
        key = RESULT_LABELS.get(self.selected_result.get())
        if self.busy or key not in self.results:
            return
        was_playing = self.playing_since is not None
        position = self.current_position()
        self.show_result(self.results[key], self.result_envelopes.get(key), store=False,
                         restore_settings=False, position=position)
        if was_playing:
            self.play_selected()

    def redraw_staff(self, validate_tempo: bool = True) -> None:
        if not self.result:
            return
        try:
            bpm = float(self.staff_bpm.get()) if validate_tempo else self.staff_tempo
            self.staff_axis, self.staff_page, self.staff_pages, self.staff_window = draw_staff(
                self.staff_figure, self.result, bpm, self.staff_page, names=self.staff_names.get(),
                event_index=self.prediction_index)
        except (ValueError, tk.TclError) as error:
            messagebox.showerror("Notation tempo", str(error), parent=self.root)
            return
        self.staff_tempo = bpm
        self.staff_line = self.staff_axis.axvline(0, color="#e15759", linewidth=1, alpha=0.8)
        local_beat = (self.cursor_seconds - self.staff_window[0]) * bpm / 60
        self.staff_line.set_xdata([local_beat, local_beat])
        self.staff_line.set_visible(0 <= local_beat <= 8)
        self.staff_cursor.set_artists([self.staff_line])
        self._staff_dirty = False
        self.staff_page_text.set(f"Page {self.staff_page + 1} / {self.staff_pages}")
        if self.tabs.select() == str(self.staff_tab):
            self.staff_canvas.draw()
        self.staff_toolbar.update()

    def change_staff_page(self, direction: int) -> None:
        self.staff_page = max(0, min(self.staff_page + direction, self.staff_pages - 1))
        self.redraw_staff()

    def on_staff_click(self, event) -> None:
        if event.inaxes is self.staff_axis and event.xdata is not None and not self.staff_toolbar.mode:
            self.seek_to(self.staff_window[0] + max(0, event.xdata) * 60 / self.staff_tempo)

    def move_cursor(self, seconds: float) -> None:
        if not self.result:
            return
        seconds = max(0.0, min(seconds, self.result["duration"]))
        self.cursor_seconds = seconds
        for line in self.cursor_lines:
            line.set_xdata([seconds, seconds])
        notes, chords = self.prediction_index.active_at(seconds)
        self.cursor_text.set(f"{seconds:.2f} s   |   Notes: {', '.join(dict.fromkeys(notes)) or 'none'}   |   Chord: {', '.join(dict.fromkeys(chords)) or 'none'}")
        if self.staff_axis is not None:
            bpm = self.staff_tempo
            target_page = min(self.staff_pages - 1, int(seconds / (8 * 60 / bpm)))
            if target_page != self.staff_page:
                self.staff_page = target_page
                self._staff_dirty = True
            local_beat = (seconds - self.staff_window[0]) * bpm / 60
            self.staff_line.set_xdata([local_beat, local_beat])
            self.staff_line.set_visible(0 <= local_beat <= 8)
        self.update_visible_cursor()

    def update_visible_cursor(self):
        selected = self.tabs.select()
        if selected == str(self.staff_tab):
            if self._staff_dirty:
                self.redraw_staff(validate_tempo=False)
            else:
                self.staff_cursor.update()
        elif selected == str(self.timeline_tab):
            self.timeline_cursor.update()

    def on_view_change(self, _event=None):
        if self.result:
            self.update_visible_cursor()

    def on_plot_click(self, event) -> None:
        if event.inaxes in self.axes and event.xdata is not None and not self.toolbar.mode:
            self.seek_to(event.xdata)

    def on_note_select(self, _event) -> None:
        selection = self.notes_table.selection()
        if selection and self.result:
            self.seek_to(self.result["notes"][int(selection[0])]["start"])

    def on_chord_select(self, _event) -> None:
        selection = self.chords_table.selection()
        if selection and self.result:
            self.seek_to(self.result["chords"][int(selection[0])]["start"])

    def open_json(self) -> None:
        if self.busy:
            messagebox.showinfo("Analysis running", "Wait for analysis to finish or cancel it first.", parent=self.root)
            return
        selected = filedialog.askopenfilename(parent=self.root, title="Open prediction JSON",
                                              initialdir=PROJECT, filetypes=[("Prediction JSON", "*.json")])
        if not selected:
            return
        try:
            result = read_prediction(selected)
            audio = Path(result.get("audio", ""))
            envelope = None
            if audio.is_file():
                try:
                    envelope = waveform_envelope(audio)
                except (ValueError, wave.Error):
                    pass
            self.clear_results()
            self.audio.set(str(audio) if audio.is_file() else "")
            self.show_result(result, envelope)
            self.status.set(f"Opened {Path(selected).name}")
        except (OSError, ValueError, KeyError, TypeError) as error:
            messagebox.showerror("Cannot open result", str(error), parent=self.root)

    def export_json(self) -> None:
        if not self.result:
            messagebox.showinfo("No result", "Analyze audio or open a result first.", parent=self.root)
            return
        selected = filedialog.asksaveasfilename(parent=self.root, title="Export prediction",
                                               defaultextension=".json", filetypes=[("JSON", "*.json")])
        if selected:
            try:
                Path(selected).write_text(json.dumps(self.result, indent=2) + "\n", encoding="utf-8")
                self.status.set(f"Saved {selected}")
            except OSError as error:
                messagebox.showerror("Cannot save result", str(error), parent=self.root)

    def export_plot(self) -> None:
        if not self.result:
            messagebox.showinfo("No result", "Analyze audio or open a result first.", parent=self.root)
            return
        selected = filedialog.asksaveasfilename(parent=self.root, title="Save current view",
                                               defaultextension=".png", filetypes=[("PNG image", "*.png")])
        if selected:
            try:
                figure = self.staff_figure if self.tabs.select() == str(self.staff_tab) else self.figure
                figure.savefig(selected, dpi=180)
                self.status.set(f"Saved {selected}")
            except OSError as error:
                messagebox.showerror("Cannot save plot", str(error), parent=self.root)

    def play(self) -> None:
        if self.busy or self.player is None:
            return
        self._halt_playback()
        self.playback_source.set("Original recording")
        self.play_selected()

    def play_selected(self):
        if self.busy or self.player is None:
            return
        source = self.playback_source.get()
        if source != "Original recording":
            key = RESULT_LABELS.get(source)
            if key not in self.results:
                messagebox.showinfo("No result", f"Analyze {source} first.", parent=self.root)
                return
            if self.result_kind != key:
                self.show_result(self.results[key], self.result_envelopes.get(key), store=False,
                                 restore_settings=False, position=self.current_position())
            self.prepare_result_audio()
            return
        audio = Path(self.audio.get())
        if not audio.is_file():
            messagebox.showinfo("Audio unavailable", "The original WAV file is unavailable for this result.", parent=self.root)
            return
        self.start_playback(audio, "original recording")

    def toggle_play(self):
        if self.busy:
            return
        if self.playing_since is not None:
            self.pause()
        else:
            self.play_selected()

    def pause(self):
        self._halt_playback()
        self.update_position(self.playback_position)
        self.status.set("Paused. Press Play to continue from this position.")

    def change_source(self, _event=None):
        if self.busy:
            return
        source = self.playback_source.get()
        was_playing = self.playing_since is not None
        self._halt_playback()
        position = self.playback_position
        key = RESULT_LABELS.get(source)
        if key in self.results:
            self.show_result(self.results[key], self.result_envelopes.get(key), store=False,
                             restore_settings=False, position=position)
        self.playback_source.set(source)
        if was_playing:
            self.play_selected()

    @staticmethod
    def clock_text(seconds):
        seconds = max(0, int(seconds))
        return f"{seconds // 60}:{seconds % 60:02d}"

    def configure_duration(self, duration):
        self.playing_duration = float(duration)
        self.seek_slider.configure(to=max(0.001, duration), state="normal" if duration and not self.busy else "disabled")
        self.position_text.set(f"{self.clock_text(self.playback_position)} / {self.clock_text(duration)}")

    def current_position(self):
        if self.playing_since is None:
            return self.playback_position
        return min(self.playing_duration, self.playback_position + time.monotonic() - self.playing_since)

    def update_position(self, seconds):
        seconds = max(0.0, min(float(seconds), self.playing_duration))
        if not self.seeking:
            self.seek_value.set(seconds)
        self.position_text.set(f"{self.clock_text(seconds)} / {self.clock_text(self.playing_duration)}")
        self.move_cursor(seconds)

    def begin_seek(self, _event=None):
        if self.busy or not self.playing_duration:
            return
        self.seek_was_playing = self.playing_since is not None
        self._halt_playback()
        self.seeking = True

    def preview_seek(self, value):
        if self.seeking:
            self.playback_position = float(value)
            self.position_text.set(f"{self.clock_text(self.playback_position)} / {self.clock_text(self.playing_duration)}")
            if self._seek_preview_after is None:
                self._seek_preview_after = self.root.after(PLAYBACK_REFRESH_MS, self.flush_seek_preview)

    def flush_seek_preview(self):
        self._seek_preview_after = None
        if self.seeking:
            self.update_position(self.playback_position)

    def end_seek(self, _event=None):
        if not self.seeking:
            return
        if self._seek_preview_after is not None:
            self.root.after_cancel(self._seek_preview_after)
            self._seek_preview_after = None
        self.seeking = False
        self.playback_position = float(self.seek_value.get())
        self.update_position(self.playback_position)
        if self.seek_was_playing:
            self.play_selected()

    def keyboard_seek(self, event=None):
        if not self.busy and (event is None or event.keysym in ("Left", "Right", "Up", "Down", "Home", "End", "Prior", "Next")):
            self.seek_to(self.seek_value.get())

    def seek_to(self, seconds):
        if self.busy:
            return
        was_playing = self.playing_since is not None
        self._halt_playback()
        self.playback_position = max(0.0, min(float(seconds), self.playing_duration))
        self.update_position(self.playback_position)
        if was_playing:
            self.play_selected()

    def start_playback(self, audio: Path, label: str) -> None:
        if self.player is None:
            return
        try:
            self._halt_playback()
            position = self.playback_position
            if self.playing_duration and position >= self.playing_duration:
                position = 0.0
            self.player.set_volume(round(self.volume.get()))
            self.player.set_gain(round(self.gain.get(), 1))
            duration = self.player.play(audio, start_seconds=position)
            self.configure_duration(self.result["duration"] if self.result else duration)
            self.playback_position = position
            self.playback_path = audio
            self.playing_since = time.monotonic()
            self.play_result_button.configure(text="Pause")
            self.update_position(position)
            self.status.set(f"Playing {label} · {piano_sound_name() if label != 'original recording' else 'source audio'}")
        except (RuntimeError, OSError, ValueError, wave.Error) as error:
            messagebox.showerror("Cannot play audio", str(error), parent=self.root)

    def update_volume(self, value) -> None:
        percent = max(0, min(100, round(float(value))))
        self.volume_text.set(f"{percent}%")
        if self.player is not None:
            self.player.set_volume(percent)

    def update_gain(self, value) -> None:
        decibels = max(0, min(24, round(float(value), 1)))
        self.gain_text.set(f"+{decibels:.1f} dB")
        if self.player is not None:
            self.player.set_gain(decibels)

    def play_result(self) -> None:
        self.playback_source.set("ByteDance" if self.result_kind == "bytedance" else "Your model")
        self.prepare_result_audio()

    def export_audio(self) -> None:
        if not self.result:
            messagebox.showinfo("No result", "Analyze audio or open a result first.", parent=self.root)
            return
        selected = filedialog.asksaveasfilename(parent=self.root, title="Export synthesized detected notes",
                                               defaultextension=".wav", filetypes=[("PCM WAV", "*.wav")])
        if selected:
            self.prepare_result_audio(Path(selected))

    def prepare_result_audio(self, export: Path | None = None) -> None:
        if self.busy:
            return
        if not self.result:
            messagebox.showinfo("No result", "Analyze audio or open a result first.", parent=self.root)
            return
        self._halt_playback()
        self.rendered_result = self.audio_cache.get(self.result_kind)
        if self.rendered_result and self.rendered_result.exists():
            if export:
                try:
                    shutil.copyfile(self.rendered_result, export)
                    self.status.set(f"Saved synthesized result audio: {export}")
                except OSError as error:
                    messagebox.showerror("Cannot save audio", str(error), parent=self.root)
            else:
                self.start_playback(self.rendered_result, self.playback_source.get())
            return
        try:
            if self.audio_temp is None:
                self.audio_temp = tempfile.TemporaryDirectory(prefix=".piano-playback-", dir=PROJECT,
                                                              ignore_cleanup_errors=True)
            path = Path(self.audio_temp.name) / f"{self.result_kind}.wav"
        except OSError as error:
            messagebox.showerror("Cannot prepare audio", str(error), parent=self.root)
            return
        self.audio_cancel.clear()
        self.set_busy(True, "audio")
        self.progress["value"] = 0
        self.status.set(f"Rendering detected notes with {piano_sound_name()}…")
        self.audio_thread = threading.Thread(target=self.audio_worker, args=(self.result, path, export, self.result_kind), daemon=True)
        self.audio_thread.start()

    def audio_worker(self, result: dict, path: Path, export: Path | None, result_kind: str) -> None:
        try:
            render_result_wav(result, path, cancel=self.audio_cancel,
                              progress=lambda done, total: self.messages.put(("audio_progress", (done, total))))
            if self.audio_cancel.is_set():
                raise InterruptedError("Result audio rendering cancelled.")
            if export:
                shutil.copyfile(path, export)
            self.messages.put(("audio_ready", (path, export, result_kind)))
        except InterruptedError:
            self.messages.put(("cancelled", None))
        except Exception as error:
            self.messages.put(("error", str(error)))

    def _halt_playback(self):
        self.playback_position = self.current_position()
        if self.player is not None:
            self.player.stop()
        self.playing_since = None
        self.play_result_button.configure(text="Play")

    def stop(self) -> None:
        if self.busy and self.task_kind == "audio":
            self.audio_cancel.set()
        self._halt_playback()
        self.playback_position = 0.0
        self.update_position(0)

    def close(self) -> None:
        self.cancel_event.set()
        self.audio_cancel.set()
        self.stop()
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=2)
        if self.audio_temp and Path(self.audio_temp.name).resolve().is_relative_to(PROJECT.resolve()):
            self.audio_temp.cleanup()
        self.root.after_cancel(self._poll_after)
        if self._seek_preview_after is not None:
            self.root.after_cancel(self._seek_preview_after)
        self.timeline_cursor.close()
        self.staff_cursor.close()
        # Matplotlib schedules idle redraws on its Tk canvas. Release those
        # callbacks before destroying the widgets (also used by UI tests).
        for canvas in (self.canvas, self.staff_canvas):
            if canvas._idle_draw_id:
                canvas.get_tk_widget().after_cancel(canvas._idle_draw_id)
                canvas._idle_draw_id = None
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    PianoApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
