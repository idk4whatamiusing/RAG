// Audio worklet for the voice input: raw PCM 16kHz Int16 -> main thread.

// capture: connect AudioWorkletNode to the same context whose sample rate is 16k
// (AudioContext({sampleRate: 16000})) so no resampling is needed.

registerProcessor("pcm16", class extends AudioWorkletProcessor {
  constructor() {
    super();
    this.acc = new Int16Array(0);
  }
  push(frame) {
    const tmp = new Int16Array(this.acc.length + frame.length);
    tmp.set(this.acc);
    for (let i = 0; i < frame.length; i++) {
      tmp[this.acc.length + i] = Math.max(-1, Math.min(1, frame[i])) * 0x7fff;
    }
    this.acc = tmp;
    // every ~100ms of audio, flush to the main thread
    if (this.acc.length >= 1600) {
      this.port.postMessage({ type: "pcm", data: this.acc.buffer, op: "audio" });
      this.acc = new Int16Array(0);
    }
  }
  process(inputs) {
    const ch = inputs[0]?.[0];
    if (ch && ch.length) this.push(ch);
    return true;
  }
});