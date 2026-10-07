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
import torch
from .thresholds import Thresholds, saved_thresholds
from .score import draw_staff
from .synthesis import render_result_wav, piano_sound_name
from .playback import WavPlayer, playback_available
from .viewer import active_at, draw_prediction, read_prediction, waveform_envelope

PROJECT = Path(__file__).resolve().parent.parent
OVERLAP_METHODS = {"Weighted average": "weighted", "50/50 average": "equal", "Keep center": "crop"}


class PianoApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("Piano Notes — Model Viewer")
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
        self.status = tk.StringVar(value="Choose a model and a piano WAV, then select Analyze.")
        self.summary = tk.StringVar(value="No analysis loaded")
        self.cursor_text = tk.StringVar(value="Click the timeline to inspect notes at a moment.")
        self.model_paths: dict[str, Path] = {}
        self.result = None
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
        self.cursor_seconds = 0
        self.audio_cancel = threading.Event()
        self.audio_temp = None
        self.audio_thread = None
        self.rendered_result = None
        self.task_kind = None
        self._build()
        self.refresh_models()
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(100, self.poll)

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
        ttk.Label(outer, text="Piano Notes", style="Title.TLabel").pack(anchor="w")
        ttk.Label(outer, text="Explore detected keys, confidence and chords across your recording.").pack(anchor="w", pady=(3, 14))
        controls = ttk.Frame(outer)
        controls.pack(fill="x")
        controls.columnconfigure(1, weight=1)
        ttk.Label(controls, text="Model").grid(row=0, column=0, sticky="w", padx=(0, 10))
        self.models_box = ttk.Combobox(controls, textvariable=self.model, state="readonly")
        self.models_box.grid(row=0, column=1, sticky="ew", pady=4)
        self.models_box.bind("<<ComboboxSelected>>", lambda _event: self.preview_thresholds())
        ttk.Button(controls, text="Browse model…", command=self.browse_model).grid(row=0, column=2, padx=8)
        ttk.Button(controls, text="Refresh", command=self.refresh_models).grid(row=0, column=3)
        ttk.Label(controls, text="Audio").grid(row=1, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.audio).grid(row=1, column=1, sticky="ew", pady=4)
        ttk.Button(controls, text="Choose WAV…", command=self.browse_audio).grid(row=1, column=2, padx=8)
        ttk.Button(controls, text="Open result JSON…", command=self.open_json).grid(row=1, column=3)
        threshold_options = ttk.Frame(outer)
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
        ttk.Label(outer, textvariable=self.threshold_details, wraplength=1150).pack(anchor="w", pady=(3, 0))
        overlap_options = ttk.Frame(outer)
        overlap_options.pack(fill="x", pady=(8, 0))
        ttk.Label(overlap_options, text="Analysis overlap (s)").pack(side="left")
        self.overlap_box = ttk.Combobox(overlap_options, values=(0, 1, 2, 3, 4),
                                        textvariable=self.overlap_seconds, state="readonly", width=5)
        self.overlap_box.pack(side="left", padx=8)
        ttk.Label(overlap_options, text="Shared audio between 4-second sections; more overlap takes longer.",
                  wraplength=600).pack(side="left")
        blend_options = ttk.Frame(outer)
        blend_options.pack(fill="x", pady=(6, 0))
        ttk.Label(blend_options, text="Overlap method").pack(side="left")
        self.overlap_method_box = ttk.Combobox(blend_options, values=tuple(OVERLAP_METHODS),
                                               textvariable=self.overlap_method, state="readonly", width=18)
        self.overlap_method_box.pack(side="left", padx=8)
        ttk.Label(blend_options, text="Weighted blends gradually; 50/50 gives both calculations equal weight.",
                  wraplength=550).pack(side="left")
        options = ttk.Frame(outer)
        options.pack(fill="x", pady=(10, 10))
        ttk.Label(options, text="Device").pack(side="left")
        ttk.Combobox(options, values=("auto", "cpu", "cuda"), textvariable=self.device,
                     state="readonly", width=7).pack(side="left", padx=8)
        self.analyze_button = ttk.Button(options, text="Analyze", style="Accent.TButton", command=self.analyze)
        self.analyze_button.pack(side="left", padx=10)
        self.cancel_button = ttk.Button(options, text="Cancel", command=self.cancel_event.set, state="disabled")
        self.cancel_button.pack(side="left")
        ttk.Button(options, text="Export JSON…", command=self.export_json).pack(side="right")
        ttk.Button(options, text="Save view…", command=self.export_plot).pack(side="right", padx=8)
        ttk.Label(outer, textvariable=self.summary, font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(3, 8))
        tabs = self.tabs = ttk.Notebook(outer)
        staff_tab = self.staff_tab = ttk.Frame(tabs)
        timeline, notes_tab, chords_tab = (ttk.Frame(tabs) for _ in range(3))
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
        self.staff_toolbar = NavigationToolbar2Tk(self.staff_canvas, staff_tab, pack_toolbar=False)
        self.staff_toolbar.pack(side="bottom", fill="x")
        self.staff_canvas.get_tk_widget().configure(height=750)
        self.staff_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.staff_canvas.mpl_connect("button_press_event", self.on_staff_click)
        self.figure = Figure(figsize=(11, 6), dpi=100)
        self.figure.text(0.5, 0.5, "Load audio and analyze, or open a saved result",
                         ha="center", va="center", color="#667085", fontsize=14)
        self.canvas = FigureCanvasTkAgg(self.figure, master=timeline)
        toolbar_row = ttk.Frame(timeline)
        toolbar_row.pack(side="bottom", fill="x")
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_row, pack_toolbar=False)
        self.toolbar.pack(side="left")
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("button_press_event", self.on_plot_click)
        self.notes_table = self._table(notes_tab, ("Note", "MIDI", "Start (s)", "End (s)", "Duration (s)", "Confidence"))
        self.chords_table = self._table(chords_tab, ("Chord", "Start (s)", "End (s)", "Duration (s)"))
        self.notes_table.bind("<<TreeviewSelect>>", self.on_note_select)
        self.chords_table.bind("<<TreeviewSelect>>", self.on_chord_select)
        playback = self.playback_controls = ttk.Frame(self.root, padding=(18, 8, 18, 0))
        playback.grid(row=1, column=0, sticky="ew")
        playback.columnconfigure(4, weight=1)
        self.play_result_button = ttk.Button(playback, text="Play result", style="Accent.TButton",
                                            command=self.play_result, state="normal" if self.player else "disabled")
        self.play_result_button.grid(row=0, column=0)
        self.play_original_button = ttk.Button(playback, text="Play original", command=self.play,
                                               state="normal" if self.player else "disabled")
        self.play_original_button.grid(row=0, column=1, padx=8)
        ttk.Button(playback, text="Stop", command=self.stop).grid(row=0, column=2)
        ttk.Label(playback, text="Volume").grid(row=0, column=3, padx=(12, 6))
        self.volume_slider = ttk.Scale(playback, from_=0, to=100, orient="horizontal", length=160,
                                       variable=self.volume, command=self.update_volume,
                                       state="normal" if self.player else "disabled")
        self.volume_slider.grid(row=0, column=4, sticky="ew")
        ttk.Label(playback, textvariable=self.volume_text, width=5, anchor="e").grid(row=0, column=5, padx=(6, 12))
        self.export_audio_button = ttk.Button(playback, text="Export result WAV…", command=self.export_audio)
        self.export_audio_button.grid(row=0, column=6)
        gain_row = ttk.Frame(playback)
        gain_row.grid(row=1, column=0, columnspan=7, sticky="ew", pady=(6, 0))
        gain_row.columnconfigure(1, weight=1)
        ttk.Label(gain_row, text="Gain").grid(row=0, column=0, padx=(0, 8))
        self.gain_slider = ttk.Scale(gain_row, from_=0, to=24, orient="horizontal", length=160,
                                     variable=self.gain, command=self.update_gain,
                                     state="normal" if self.player else "disabled")
        self.gain_slider.grid(row=0, column=1, sticky="ew")
        ttk.Label(gain_row, textvariable=self.gain_text, width=9, anchor="e").grid(row=0, column=2, padx=(8, 0))
        ttk.Label(playback, text=f"Sound: {piano_sound_name()}").grid(row=2, column=0, columnspan=7,
                                                                  sticky="w", pady=(4, 0))
        footer = ttk.Frame(self.root, padding=(18, 8, 18, 12))
        footer.grid(row=2, column=0, sticky="ew")
        ttk.Label(footer, textvariable=self.cursor_text).pack(anchor="w", pady=(0, 4))
        self.progress = ttk.Progressbar(footer, mode="determinate")
        self.progress.pack(fill="x", pady=4)
        ttk.Label(footer, textvariable=self.status, wraplength=1150).pack(anchor="w")
        tabs.pack(fill="both", expand=True, pady=(8, 0))
        self.root.bind("<MouseWheel>", self.scroll_page, add="+")
        self.root.bind("<Button-4>", self.scroll_page, add="+")
        self.root.bind("<Button-5>", self.scroll_page, add="+")

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
            preferred = ("piano-v8.pt", "piano-v7.pt", "piano-v6.pt", "piano-v5.pt", "piano-v2-separate.pt", "piano-v2-tuned.pt", "piano-v2.pt", "piano-tuned.pt", "piano.pt", "piano-demo.pt")
            first = next((name for name in preferred if name in self.model_paths), next(iter(self.model_paths)))
            self.model.set(first)
        self.preview_thresholds()

    def preview_thresholds(self):
        if self.busy or self.model.get() not in self.model_paths:
            return
        try:
            model = load_model(self.model_paths[self.model.get()], torch.device("cpu"))
            self.display_thresholds(model.decoding_thresholds)
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
        selected = filedialog.askopenfilename(parent=self.root, title="Choose piano audio",
                                              initialdir=PROJECT / "data",
                                              filetypes=[("16-bit PCM WAV", "*.wav")])
        if selected:
            self.audio.set(selected)

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
        self.stop()
        self.set_busy(True, "analysis")
        self.cancel_event.clear()
        self.progress["value"] = 0
        self.status.set(f"Loading {model.name} and analyzing {audio.name}…")
        device = self.device.get()
        threading.Thread(target=self._worker, args=(audio, model, thresholds, device, overlap, overlap_blend), daemon=True).start()

    def set_busy(self, busy: bool, kind=None) -> None:
        self.busy = busy
        self.task_kind = kind if busy else None
        self.analyze_button.configure(state="disabled" if busy else "normal")
        self.overlap_box.configure(state="disabled" if busy else "readonly")
        self.overlap_method_box.configure(state="disabled" if busy else "readonly")
        self.cancel_button.configure(state="normal" if busy else "disabled", command=self.cancel_active)
        state = "normal" if self.player and not busy else "disabled"
        self.play_result_button.configure(state=state)
        self.play_original_button.configure(state=state)
        self.export_audio_button.configure(state="disabled" if busy else "normal")

    def cancel_active(self) -> None:
        self.cancel_event.set()
        self.audio_cancel.set()
        self.status.set("Cancelling the current task…")

    def _worker(self, audio, model, thresholds, device, overlap_seconds, overlap_blend) -> None:
        try:
            result = transcribe_audio(audio, model, device=device,
                                     **{f"{head}_threshold": value for head, value in thresholds.items()},
                                     progress=lambda done, total: self.messages.put(("progress", (done, total))),
                                     cancel=self.cancel_event, context_seconds=overlap_seconds / 2,
                                     overlap_blend=overlap_blend)
            envelope = waveform_envelope(audio)
            if self.cancel_event.is_set():
                raise InterruptedError("Analysis cancelled.")
            self.messages.put(("result", (result, envelope)))
        except InterruptedError:
            self.messages.put(("cancelled", None))
        except Exception as error:
            self.messages.put(("error", str(error)))

    def poll(self) -> None:
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
                        self.show_result(*value)
                        self.status.set("Analysis complete. Inspect the Staff tab or select Play result to hear the detected notes.")
                    elif kind == "audio_ready":
                        path, export = value
                        self.rendered_result = path
                        if export:
                            self.status.set(f"Saved synthesized result audio: {export}")
                        else:
                            self.start_playback(path, "detected notes")
                    elif kind == "cancelled":
                        self.status.set("Task cancelled.")
                    else:
                        self.status.set("Task failed.")
                        messagebox.showerror("Task failed", value, parent=self.root)
        except queue.Empty:
            pass
        if self.playing_since is not None and self.result:
            elapsed = time.monotonic() - self.playing_since
            if self.player is None or not self.player.is_playing:
                self.stop()
            else:
                self.move_cursor(elapsed)
        self.root.after(100, self.poll)

    def show_result(self, result: dict, envelope=None) -> None:
        self.stop()
        self.result = result
        self.rendered_result = None
        self.cursor_seconds = 0
        self.staff_page = 0
        overlap = result.get("inference_config", {}).get("overlap_seconds")
        if overlap in (0, 1, 2, 3, 4):
            self.overlap_seconds.set(overlap)
        blend = result.get("inference_config", {}).get("overlap_blend", "crop")
        self.overlap_method.set(next((label for label, mode in OVERLAP_METHODS.items() if mode == blend), "Keep center"))
        if result.get("model") and Path(result["model"]).is_file():
            model_path = Path(result["model"]).resolve()
            label = next((name for name, path in self.model_paths.items() if path.resolve() == model_path), str(model_path))
            self.model_paths[label] = model_path
            self.models_box["values"] = list(self.model_paths)
            self.model.set(label)
        thresholds = saved_thresholds(result)
        self.display_thresholds(thresholds)
        self.axes = draw_prediction(self.figure, result, envelope)
        self.cursor_lines = [axis.axvline(0, color="#e15759", linewidth=1, alpha=0.8) for axis in self.axes]
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
        threshold_label = "pitch-dependent thresholds" if thresholds.pitch_dependent else (
            f"frame/onset/offset {summary['frame']:.2f}/{summary['onset']:.2f}/{summary['offset']:.2f}")
        self.summary.set(f"{result['duration']:.2f} seconds   ·   {len(result['notes'])} note events   ·   "
                         f"{len(result['chords'])} chord spans   ·   {model_name}   ·   "
                         + threshold_label)
        self.redraw_staff(validate_tempo=False)
        self.tabs.select(self.staff_tab)
        self.move_cursor(0)

    def redraw_staff(self, validate_tempo: bool = True) -> None:
        if not self.result:
            return
        try:
            bpm = float(self.staff_bpm.get()) if validate_tempo else self.staff_tempo
            self.staff_axis, self.staff_page, self.staff_pages, self.staff_window = draw_staff(
                self.staff_figure, self.result, bpm, self.staff_page, names=self.staff_names.get())
        except (ValueError, tk.TclError) as error:
            messagebox.showerror("Notation tempo", str(error), parent=self.root)
            return
        self.staff_tempo = bpm
        self.staff_line = self.staff_axis.axvline(0, color="#e15759", linewidth=1, alpha=0.8)
        local_beat = (self.cursor_seconds - self.staff_window[0]) * bpm / 60
        self.staff_line.set_xdata([local_beat, local_beat])
        self.staff_line.set_visible(0 <= local_beat <= 8)
        self.staff_page_text.set(f"Page {self.staff_page + 1} / {self.staff_pages}")
        self.staff_canvas.draw()
        self.staff_toolbar.update()

    def change_staff_page(self, direction: int) -> None:
        self.staff_page = max(0, min(self.staff_page + direction, self.staff_pages - 1))
        self.redraw_staff()

    def on_staff_click(self, event) -> None:
        if event.inaxes is self.staff_axis and event.xdata is not None and not self.staff_toolbar.mode:
            self.move_cursor(self.staff_window[0] + max(0, event.xdata) * 60 / self.staff_tempo)

    def move_cursor(self, seconds: float) -> None:
        if not self.result:
            return
        seconds = max(0.0, min(seconds, self.result["duration"]))
        self.cursor_seconds = seconds
        for line in self.cursor_lines:
            line.set_xdata([seconds, seconds])
        notes, chords = active_at(self.result, seconds)
        self.cursor_text.set(f"{seconds:.2f} s   |   Notes: {', '.join(notes) or 'none'}   |   Chord: {', '.join(chords) or 'none'}")
        self.canvas.draw_idle()
        if self.staff_axis is not None:
            start, end = self.staff_window
            bpm = self.staff_tempo
            if not start <= seconds < end:
                self.staff_page = min(self.staff_pages - 1, int(seconds / (8 * 60 / bpm)))
                self.redraw_staff(validate_tempo=False)
            local_beat = (seconds - self.staff_window[0]) * bpm / 60
            self.staff_line.set_xdata([local_beat, local_beat])
            self.staff_line.set_visible(0 <= local_beat <= 8)
            self.staff_canvas.draw_idle()

    def on_plot_click(self, event) -> None:
        if event.inaxes in self.axes and event.xdata is not None and not self.toolbar.mode:
            self.move_cursor(event.xdata)

    def on_note_select(self, _event) -> None:
        selection = self.notes_table.selection()
        if selection and self.result:
            self.move_cursor(self.result["notes"][int(selection[0])]["start"])

    def on_chord_select(self, _event) -> None:
        selection = self.chords_table.selection()
        if selection and self.result:
            self.move_cursor(self.result["chords"][int(selection[0])]["start"])

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
        if not self.result or self.player is None:
            return
        audio = Path(self.result.get("audio", ""))
        if not audio.is_file():
            messagebox.showinfo("Audio unavailable", "The original WAV file is unavailable for this result.", parent=self.root)
            return
        self.start_playback(audio, "original recording")

    def start_playback(self, audio: Path, label: str) -> None:
        if self.player is None:
            return
        try:
            self.stop()
            self.player.set_volume(round(self.volume.get()))
            self.player.set_gain(round(self.gain.get(), 1))
            self.playing_duration = self.player.play(audio)
            self.playing_since = time.monotonic()
            self.move_cursor(0)
            self.status.set(f"Playing {label} from the start.")
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
        if not self.result or not self.result["notes"]:
            messagebox.showinfo("No detected notes", "Analyze audio or open a result containing notes first.", parent=self.root)
            return
        self.stop()
        if self.rendered_result and self.rendered_result.exists():
            if export:
                try:
                    shutil.copyfile(self.rendered_result, export)
                    self.status.set(f"Saved synthesized result audio: {export}")
                except OSError as error:
                    messagebox.showerror("Cannot save audio", str(error), parent=self.root)
            else:
                self.start_playback(self.rendered_result, "detected notes")
            return
        try:
            if self.audio_temp is None:
                self.audio_temp = tempfile.TemporaryDirectory(prefix=".piano-playback-", dir=PROJECT,
                                                              ignore_cleanup_errors=True)
            path = Path(self.audio_temp.name) / "result.wav"
        except OSError as error:
            messagebox.showerror("Cannot prepare audio", str(error), parent=self.root)
            return
        self.audio_cancel.clear()
        self.set_busy(True, "audio")
        self.progress["value"] = 0
        self.status.set(f"Rendering detected notes with {piano_sound_name()}…")
        self.audio_thread = threading.Thread(target=self.audio_worker, args=(self.result, path, export), daemon=True)
        self.audio_thread.start()

    def audio_worker(self, result: dict, path: Path, export: Path | None) -> None:
        try:
            render_result_wav(result, path, cancel=self.audio_cancel,
                              progress=lambda done, total: self.messages.put(("audio_progress", (done, total))))
            if export:
                shutil.copyfile(path, export)
            if self.audio_cancel.is_set():
                raise InterruptedError("Result audio rendering cancelled.")
            self.messages.put(("audio_ready", (path, export)))
        except InterruptedError:
            self.messages.put(("cancelled", None))
        except Exception as error:
            self.messages.put(("error", str(error)))

    def stop(self) -> None:
        if self.busy and self.task_kind == "audio":
            self.audio_cancel.set()
        if self.player is not None:
            self.player.stop()
        self.playing_since = None

    def close(self) -> None:
        self.cancel_event.set()
        self.audio_cancel.set()
        self.stop()
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=2)
        if self.audio_temp and Path(self.audio_temp.name).resolve().is_relative_to(PROJECT.resolve()):
            self.audio_temp.cleanup()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    PianoApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
