// One HTML media element owns the clock. Pausing and seeking retain that clock.
export class AudioTransport {
  constructor(audio, {getURL, onChange = () => {}, onError = () => {}, createContext} = {}) {
    this.audio = audio;
    this.getURL = getURL;
    this.onChange = onChange;
    this.onError = onError;
    this.createContext = createContext || (() => new (globalThis.AudioContext || globalThis.webkitAudioContext)());
    this.source = "result";
    this.position = 0;
    this.expectedDuration = 0;
    this.volume = 50;
    this.gainDB = 0;
    this.wantPlaying = false;
    this.loading = false;
    this.revision = 0;
    this.readyURL = null;
    this.context = null;
    for (const event of ["timeupdate", "durationchange", "loadedmetadata", "play", "pause", "waiting", "playing", "seeked"]) {
      audio.addEventListener(event, () => {
        if (audio.readyState >= 1) this.position = audio.currentTime;
        this.publish();
      });
    }
    audio.addEventListener("ended", () => { this.wantPlaying = false; this.publish(); });
    audio.addEventListener("error", () => {
      if (!this.loading && this.readyURL) { this.pause(); this.onError(new Error("Cannot play this audio. Try preparing it again.")); }
    });
  }

  snapshot() {
    return {source: this.source, position: this.position,
      duration: Number.isFinite(this.audio.duration) ? this.audio.duration : this.expectedDuration,
      playing: this.wantPlaying || !this.audio.paused, loading: this.loading};
  }

  publish() { this.onChange(this.snapshot()); }

  select(source, duration, reset = false) {
    const position = reset ? 0 : this.position;
    this.pause();
    this.source = source;
    this.expectedDuration = duration;
    this.readyURL = null;
    this.audio.removeAttribute("src");
    this.audio.load();
    this.position = Math.max(0, Math.min(position, duration));
    this.publish();
  }

  setLevels(volume, gainDB) {
    this.volume = Math.max(0, Math.min(100, Number(volume)));
    this.gainDB = Math.max(0, Math.min(24, Number(gainDB)));
    const level = this.volume / 100 * 10 ** (this.gainDB / 20);
    if (this.gainNode) this.gainNode.gain.value = level;
    else this.audio.volume = this.volume / 100;
  }

  async enableAudio() {
    if (!this.context) {
      this.context = this.createContext();
      this.mediaNode = this.context.createMediaElementSource(this.audio);
      this.gainNode = this.context.createGain();
      this.limiter = this.context.createDynamicsCompressor();
      this.limiter.threshold.value = -3;
      this.limiter.knee.value = 6;
      this.limiter.ratio.value = 20;
      this.limiter.attack.value = 0.005;
      this.limiter.release.value = 0.15;
      this.mediaNode.connect(this.gainNode).connect(this.limiter).connect(this.context.destination);
      this.audio.volume = 1;
      this.setLevels(this.volume, this.gainDB);
    }
    await this.context.resume();
  }

  async toggle() {
    if (this.wantPlaying || !this.audio.paused) { this.pause(); return; }
    const revision = ++this.revision;
    this.wantPlaying = true;
    this.loading = true;
    this.publish();
    try {
      // Resume AudioContext immediately inside the user's click gesture.
      await this.enableAudio();
      if (!this.readyURL) {
        const url = await this.getURL(this.source);
        if (revision !== this.revision) return;
        await this.loadURL(url);
      }
      if (revision !== this.revision) return;
      if (this.audio.ended || this.position >= this.audio.duration) this.position = 0;
      this.audio.currentTime = Math.min(this.position, this.audio.duration || this.expectedDuration);
      await this.audio.play();
      if (revision !== this.revision) this.audio.pause();
    } catch (error) {
      if (revision === this.revision) { this.wantPlaying = false; this.onError(error); }
    } finally {
      if (revision === this.revision) { this.loading = false; this.publish(); }
    }
  }

  loadURL(url) {
    const target = this.position;
    return new Promise((resolve, reject) => {
      const clean = () => { this.audio.removeEventListener("loadedmetadata", loaded); this.audio.removeEventListener("error", failed); };
      const loaded = () => {
        clean(); this.readyURL = url;
        this.position = Math.min(target, this.audio.duration);
        this.audio.currentTime = this.position;
        resolve();
      };
      const failed = () => { clean(); reject(new Error("Audio could not be loaded. Choose a valid WAV or render the result again.")); };
      this.audio.addEventListener("loadedmetadata", loaded);
      this.audio.addEventListener("error", failed);
      this.audio.src = url;
      this.audio.load();
    });
  }

  pause() {
    this.revision++;
    this.wantPlaying = false;
    this.loading = false;
    this.audio.pause();
    this.publish();
  }

  seek(seconds) {
    const duration = this.readyURL && Number.isFinite(this.audio.duration) ? this.audio.duration : this.expectedDuration;
    this.position = Math.max(0, Math.min(Number(seconds), duration));
    if (this.readyURL && this.audio.readyState >= 1) this.audio.currentTime = this.position;
    this.publish();
  }

  stop() { this.pause(); this.seek(0); }
}

export function formatTime(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) seconds = 0;
  const minutes = Math.floor(seconds / 60);
  return `${minutes}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;
}
