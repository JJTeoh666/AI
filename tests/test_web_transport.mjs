import test from "node:test";
import assert from "node:assert/strict";
import {AudioTransport, formatTime} from "../piano_ml/web_static/transport.js";

class FakeAudio extends EventTarget {
  constructor() { super(); this.paused = true; this.currentTime = 0; this.duration = NaN; this.readyState = 0; this.volume = 1; this.ended = false; }
  pause() { this.paused = true; this.dispatchEvent(new Event("pause")); }
  async play() { this.paused = false; this.dispatchEvent(new Event("play")); }
  removeAttribute() { this.src = ""; }
  load() {
    this.readyState = 0; this.duration = NaN;
    if (!this.src) return;
    queueMicrotask(() => { this.duration = 12; this.readyState = 1; this.dispatchEvent(new Event("loadedmetadata")); });
  }
}
function fakeContext() {
  const node = () => ({gain: {value: 1}, threshold: {}, knee: {}, ratio: {}, attack: {}, release: {}, connect(next) { return next; }});
  return {createMediaElementSource: node, createGain: node, createDynamicsCompressor: node, destination: {}, async resume() {}};
}
function player(getURL = async () => "/piano.wav") {
  const audio = new FakeAudio();
  const transport = new AudioTransport(audio, {getURL, createContext: fakeContext});
  transport.select("result", 12, true);
  return {audio, transport};
}

test("play, pause and resume retain the audio clock", async () => {
  const {audio, transport} = player();
  await transport.toggle(); assert.equal(audio.paused, false);
  audio.currentTime = 4.2; audio.dispatchEvent(new Event("timeupdate"));
  await transport.toggle(); assert.equal(audio.paused, true); assert.equal(transport.position, 4.2);
  await transport.toggle(); assert.equal(audio.currentTime, 4.2); assert.equal(audio.paused, false);
});
test("dragging seeks while playing and paused without changing playback state", async () => {
  const {audio, transport} = player();
  await transport.toggle(); transport.seek(7.5);
  assert.equal(audio.currentTime, 7.5); assert.equal(audio.paused, false);
  transport.pause(); transport.seek(2.75);
  assert.equal(audio.currentTime, 2.75); assert.equal(audio.paused, true);
  transport.seek(30); assert.equal(audio.currentTime, 12);
  transport.stop(); assert.equal(audio.currentTime, 0); assert.equal(audio.paused, true);
});
test("seeking before piano rendering completes starts at the chosen position", async () => {
  let release;
  const {audio, transport} = player(() => new Promise(resolve => { release = resolve; }));
  const playing = transport.toggle();
  await new Promise(resolve => setImmediate(resolve));
  transport.seek(6); release("/piano.wav"); await playing;
  assert.equal(audio.currentTime, 6); assert.equal(audio.paused, false);
});
test("pause during rendering prevents late autoplay", async () => {
  let release;
  const {audio, transport} = player(() => new Promise(resolve => { release = resolve; }));
  const playing = transport.toggle(); await new Promise(resolve => setImmediate(resolve));
  transport.pause(); release("/piano.wav"); await playing;
  assert.equal(audio.paused, true); assert.equal(transport.wantPlaying, false);
});
test("source switching keeps position and pauses until Play is selected", async () => {
  const {audio, transport} = player();
  await transport.toggle(); transport.seek(5);
  transport.select("original", 10);
  assert.equal(transport.position, 5); assert.equal(audio.paused, true);
  await transport.toggle(); assert.equal(audio.currentTime, 5);
});
test("volume starts at 50 percent, gain scales it and mute stays silent", async () => {
  const {audio, transport} = player();
  transport.setLevels(50, 0); assert.equal(audio.volume, 0.5);
  await transport.toggle(); assert.equal(transport.gainNode.gain.value, 0.5);
  transport.setLevels(50, 6); assert.ok(Math.abs(transport.gainNode.gain.value - 0.997631) < 0.00001);
  transport.setLevels(0, 24); assert.equal(transport.gainNode.gain.value, 0);
});
test("time labels and end of playback", async () => {
  const {audio, transport} = player();
  assert.equal(formatTime(65.9), "1:05"); assert.equal(formatTime(NaN), "0:00");
  await transport.toggle(); audio.currentTime = 12; audio.ended = true; audio.paused = true;
  audio.dispatchEvent(new Event("timeupdate")); audio.dispatchEvent(new Event("ended"));
  assert.equal(transport.snapshot().playing, false);
  await transport.toggle(); assert.equal(audio.currentTime, 0);
});
