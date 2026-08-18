"""Audio worklet for the voice input: raw PCM 16kHz Int16 -> main thread."""

// capture: connect AudioWorkletNode to the same context whose sample rate is 16k
// (AudioContext({sampleRate: 16000})) so no resampling is needed.
let first = true;
let offset = 0;
let buf = new Int16Array(0);

function push(frame: Float32Array) {
  const tmp = new Int16Array(buf.length + frame.length);
  tmp.set(buf);
  for (let i = 0; i < frame.length; i++) {
    tmp[buf.length + i] = Math.max(-1, Math.min(1, frame[i])) * 0x7fff;
  }
  buf = tmp;
  // every ~100ms of audio, flush
  if (buf.length >= 1600) {
    postMessage({ type: "pcm", data: buf.buffer, op: "audio" });
    buf = new Int16Array(0);
  }
}

registerProcessor("pcm16", class extends AudioWorkletProcessor {
  process(inputs: Float32Array[][]) {
    const ch = inputs[0]?.[0];
    if (ch && ch.length) push(ch);
    return true;
  }
});