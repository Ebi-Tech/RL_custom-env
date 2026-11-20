const MASTER_LEVEL = 0.5;

function noiseBuffer(context, seconds) {
  /** Return a buffer of soft brown noise, the base of the crowd murmur. */
  const length = Math.floor(context.sampleRate * seconds);
  const buffer = context.createBuffer(1, length, context.sampleRate);
  const data = buffer.getChannelData(0);
  let last = 0;
  for (let i = 0; i < length; i += 1) {
    last = (last + 0.02 * (Math.random() * 2 - 1)) / 1.02;
    data[i] = last * 3.5;
  }
  return buffer;
}

function murmur(context, output, buffer) {
  /** Start the crowd bed: looped noise through voice-like band filters with a slow swell. */
  const source = context.createBufferSource();
  source.buffer = buffer;
  source.loop = true;
  const gain = context.createGain();
  gain.gain.value = 0.0;
  for (const [frequency, q] of [[420, 1.2], [900, 2.5], [1700, 3]]) {
    const band = context.createBiquadFilter();
    band.type = "bandpass";
    band.frequency.value = frequency;
    band.Q.value = q;
    source.connect(band).connect(gain);
  }
  const swell = context.createOscillator();
  const depth = context.createGain();
  swell.frequency.value = 0.13;
  depth.gain.value = 0.25;
  swell.connect(depth).connect(gain.gain);
  gain.connect(output);
  source.start();
  swell.start();
  return gain;
}

function voice(context, output, loudness) {
  /** Play one distant call or chat fragment: a short gliding tone through a vowel filter. */
  const now = context.currentTime;
  const tone = context.createOscillator();
  tone.type = "sawtooth";
  const base = 140 + Math.random() * 160;
  tone.frequency.setValueAtTime(base, now);
  tone.frequency.linearRampToValueAtTime(base * (0.8 + Math.random() * 0.5), now + 0.5);
  const vowel = context.createBiquadFilter();
  vowel.type = "bandpass";
  vowel.frequency.value = 600 + Math.random() * 900;
  vowel.Q.value = 6;
  const gain = context.createGain();
  gain.gain.setValueAtTime(0, now);
  gain.gain.linearRampToValueAtTime(0.05 * loudness, now + 0.06);
  gain.gain.linearRampToValueAtTime(0, now + 0.35 + Math.random() * 0.4);
  const pan = context.createStereoPanner();
  pan.pan.value = Math.random() * 2 - 1;
  tone.connect(vowel).connect(gain).connect(pan).connect(output);
  tone.start(now);
  tone.stop(now + 1);
}

function motoPass(context, output) {
  /** Play a moto taxi passing by: a buzzing engine that rises, pans across and fades. */
  const now = context.currentTime;
  const engine = context.createOscillator();
  engine.type = "sawtooth";
  engine.frequency.setValueAtTime(70, now);
  engine.frequency.linearRampToValueAtTime(95, now + 1.4);
  engine.frequency.linearRampToValueAtTime(65, now + 3);
  const filter = context.createBiquadFilter();
  filter.type = "lowpass";
  filter.frequency.value = 700;
  const gain = context.createGain();
  gain.gain.setValueAtTime(0, now);
  gain.gain.linearRampToValueAtTime(0.06, now + 1.3);
  gain.gain.linearRampToValueAtTime(0, now + 3);
  const pan = context.createStereoPanner();
  pan.pan.setValueAtTime(-0.9, now);
  pan.pan.linearRampToValueAtTime(0.9, now + 3);
  engine.connect(filter).connect(gain).connect(pan).connect(output);
  engine.start(now);
  engine.stop(now + 3.1);
}

export function createSound() {
  /** Create the optional market sound, silent until switched on. */
  return { context: null, master: null, bed: null, on: false, nextVoice: 0, nextMoto: 0, busy: 0.5 };
}

export function setSound(sound, on) {
  /** Switch the market sound on or off; the audio graph is built the first time it is switched on. */
  sound.on = on;
  if (on && !sound.context) {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    if (!AudioContext) return false;
    sound.context = new AudioContext();
    sound.master = sound.context.createGain();
    sound.master.connect(sound.context.destination);
    sound.bed = murmur(sound.context, sound.master, noiseBuffer(sound.context, 4));
  }
  if (!sound.context) return on;
  if (on) sound.context.resume();
  sound.master.gain.setTargetAtTime(on ? MASTER_LEVEL : 0, sound.context.currentTime, 0.4);
  return on;
}

export function updateSound(sound, busy, time) {
  /** Follow the crowd: a louder murmur and more voices on busy days, and the odd passing moto. */
  if (!sound.on || !sound.context) return;
  sound.busy = busy;
  sound.bed.gain.setTargetAtTime(0.25 + busy * 0.45, sound.context.currentTime, 1.5);
  if (time > sound.nextVoice) {
    voice(sound.context, sound.master, 0.5 + busy);
    sound.nextVoice = time + 0.25 + Math.random() * (1.6 - busy);
  }
  if (time > sound.nextMoto) {
    motoPass(sound.context, sound.master);
    sound.nextMoto = time + 6 + Math.random() * 10;
  }
}
