// ふぇれ子 3D (VRM 1.0) for the chat screen, inside an Android WebView over the blurred background.
// Everything is served by the app itself (https://fereko.local/...), nothing from the network.
//
// The app (window.Fereko, a Java object) is polled every frame:
//   Fereko.state()  "loading" | "idle" | "listen" | "recognize" | "think" | "talk"
//   Fereko.level()  loudness of the voice being heard now (0..1)
//   Fereko.log(json) one record for runs.jsonl (load report, frame-time stats)
//   Fereko.anims()  JSON list of the .vrma files in the model folder
// and it calls:
//   window.avatarCue({emotion, actions, text, event})  a sentence starts playing (emoji → emotion,
//                                                      ［うなずく］-style tags → reactions, text → mouth shapes)
//   window.avatarInterrupt()                           the user talked over her (startle, then listen)
//   window.playReaction(name)                          (a tap on her plays "turn_around")
//
// Motion (3D受け渡し.md + motion-14 CLAUDE-HANDOFF.md):
//   base layer   state loops idle / listen / think / talk(_2/_3), cross-faded 0.3 s; think is entered and
//                left through think_enter / think_exit (0.15 s boundaries)
//   reaction     upper-body one-shots replace the base's upper body with weights summing to 1
//                (0.2 s in, 0.3 s out); legs and hips stay on the base
//   full body    turn_around replaces everything (0.3 s in/out); an interruption waits for it to end
// Clips are sampled track by track (three.js's AnimationMixer returned the rest pose for the 180-degree
// part of turn_around). Models without VRMA loops fall back to a small procedural motion.
//
// Face: emotion = <emo> (brows) + <emo>_lids × (1 − blink) + <emo>_mouth × (k while speaking, else 1),
// natural blink combined with timed cues (giggle_hide, yawn), vowels from the voice level and the kana
// of the sentence, gaze up-left while thinking. Missing expressions are skipped.
//
// 2D (?flat=<rig.json>): the same motion runs on the (undrawn) VRM's normalized bones and flat.mjs draws
// the picture layers from that pose instead. The face is not drawn in 2D yet.
import * as THREE from 'three';
import { GLTFLoader } from './vendor/GLTFLoader.js';
import { VRMLoaderPlugin } from './vendor/three-vrm.module.js';
import { VRMAnimationLoaderPlugin, createVRMAnimationClip } from './vendor/three-vrm-animation.module.js';
import { createFlat } from './flat.mjs';

const params = new URLSearchParams(location.search);
const VRM_URL = params.get('vrm');
// ?flat=<rig.json>: show the flat (2D) picture instead of the model; the VRM still runs the motions
const FLAT_URL = params.get('flat');
const PIXEL_SCALE = parseFloat(params.get('scale') || '0.75');   // of devicePixelRatio
const FPS_CAP = 60;
// framing: ears to knees between these fractions of the view height (the same as the 2D picture)
const VIEW_TOP = 0.11, VIEW_BOTTOM = 0.93;

// in a PC browser (no app): ?anims=a.vrma,b.vrma lists the motions, ?state=talk fakes the state
const app = window.Fereko || { state: () => params.get('state') || 'idle', level: () => 0, log: (s) => console.log(s),
  anims: () => JSON.stringify((params.get('anims') || '').split(',').filter(Boolean)) };
const log = (o) => { try { app.log(JSON.stringify(o)); } catch (e) { /* app gone */ } };

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
renderer.setClearColor(0x000000, 0);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.setPixelRatio(window.devicePixelRatio * PIXEL_SCALE);
renderer.setSize(innerWidth, innerHeight);
document.body.append(renderer.domElement);

const scene = new THREE.Scene();
scene.add(new THREE.HemisphereLight(0xffffff, 0xb8bccc, 1.1));
const sun = new THREE.DirectionalLight(0xffffff, 0.9);
sun.position.set(2, 4, 3);   // model-reference/lighting-reference.json
scene.add(sun);
const camera = new THREE.PerspectiveCamera(24, innerWidth / innerHeight, 0.05, 20);

let vrm = null, span = null;   // span: {top, bottom} world y of ears and knees
let flat = null;

function frame() {
  if (!span) return;
  // the visible height H covers ears..knees from VIEW_TOP to VIEW_BOTTOM of the screen
  const H = (span.top - span.bottom) / (VIEW_BOTTOM - VIEW_TOP);
  const screenTop = span.top + VIEW_TOP * H;
  const centerY = screenTop - H / 2;
  const fov = THREE.MathUtils.degToRad(camera.fov);
  const dist = (H / 2) / Math.tan(fov / 2);
  camera.aspect = innerWidth / innerHeight;
  camera.position.set(0, centerY, dist);
  camera.lookAt(0, centerY, 0);
  camera.updateProjectionMatrix();
}

addEventListener('resize', () => {
  renderer.setSize(innerWidth, innerHeight);
  frame();
  if (flat) flat.frame(innerWidth / innerHeight, VIEW_TOP, VIEW_BOTTOM);
});
const draw = () => (flat ? renderer.render(flat.scene, flat.camera) : renderer.render(scene, camera));

const bone = (name) => vrm && vrm.humanoid.getNormalizedBoneNode(name);
const setRot = (name, x, y, z) => { const b = bone(name); if (b) b.rotation.set(x, y, z); };
const has = (name) => !!(vrm && vrm.expressionManager && vrm.expressionManager.getExpression(name));
const setExpr = (name, w) => { if (has(name)) vrm.expressionManager.setValue(name, Math.max(0, Math.min(1, w))); };
const clamp01 = (x) => Math.max(0, Math.min(1, x));

async function load() {
  const t0 = performance.now();
  const loader = new GLTFLoader();
  loader.register((p) => new VRMLoaderPlugin(p));
  const gltf = await loader.loadAsync(VRM_URL);
  vrm = gltf.userData.vrm;
  if (!vrm) throw new Error('VRM not found in file');
  if (!FLAT_URL) scene.add(vrm.scene);
  vrm.scene.rotation.y = 0;   // VRM 1.0 faces +Z, towards the camera
  if (vrm.lookAt) vrm.lookAt.autoUpdate = false;   // the gaze expressions are set here (no double application)
  const nroot = vrm.humanoid.normalizedHumanBonesRoot;
  if (nroot && !vrm.scene.getObjectByName('Normalized_hips')) vrm.scene.add(nroot);
  await loadMotions();
  // first pose, then humanoid update, world matrices, spring reset (motion-14 handoff: not every frame)
  vrm.humanoid.resetNormalizedPose();
  if (clips.idle) writeClip(clips.idle, 0, 1, null); else poseRest();
  vrm.update(0);
  vrm.scene.updateMatrixWorld(true);
  vrm.springBoneManager?.reset();
  if (FLAT_URL) {
    const f = await createFlat(FLAT_URL);
    f.setReference(flatLocal, bone('hips').position, bone('hips').position.y);
    f.frame(innerWidth / innerHeight, VIEW_TOP, VIEW_BOTTOM);
    flat = f;
  }

  // framing from the model itself: top of the ears, and the knees
  const box = new THREE.Box3().setFromObject(vrm.scene);
  const knee = new THREE.Vector3();
  (bone('leftLowerLeg') || vrm.scene).getWorldPosition(knee);
  span = { top: box.max.y, bottom: knee.y };
  frame();

  // load report (what this model can and can't do)
  const json = gltf.parser.json;
  const ext = json.extensions?.VRMC_vrm || {};
  const exprs = vrm.expressionManager ? vrm.expressionManager.expressions : [];
  const functional = exprs.filter((e) => (e._binds?.length || 0) > 0).map((e) => e.expressionName);
  const wanted = ['blink', 'aa', 'ih', 'ou', 'ee', 'oh', 'jito', 'smug', 'sleepy', 'giggle', 'sigh', 'surprised', 'happy'];
  let tris = 0;
  for (const m of json.meshes || []) for (const p of m.primitives) tris += (json.accessors[p.indices]?.count || 0) / 3;
  const gl = renderer.getContext();
  const dbg = gl.getExtension('WEBGL_debug_renderer_info');
  log({
    kind: 'avatar_load',
    motions: Object.keys(clips),
    ms: Math.round(performance.now() - t0),
    modelVersion: ext.meta?.version, authors: ext.meta?.authors,
    triangles: tris,
    images: (json.images || []).length,
    functionalExpressions: functional.length,
    missingFunctional: wanted.filter((n) => !functional.includes(n)),
    lookAtType: json.extensions?.VRMC_vrm?.lookAt?.type || null,
    springJoints: vrm.springBoneManager ? vrm.springBoneManager.joints.size : 0,
    span: { top: +span.top.toFixed(3), bottom: +span.bottom.toFixed(3) },
    flat: !!flat,
    three: THREE.REVISION,
    gpu: dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : null,
    ua: navigator.userAgent,
    pixelRatio: renderer.getPixelRatio(), canvas: [renderer.domElement.width, renderer.domElement.height],
  });
}

const flatLocal = (name) => { const b = bone(name); return b ? b.quaternion : null; };

function poseRest() {
  // T pose -> arms down along the body, elbows slightly bent (models without VRMA loops)
  setRot('leftUpperArm', 0, 0, -1.15);
  setRot('rightUpperArm', 0, 0, 1.15);
  setRot('leftLowerArm', 0, 0.1, 0);
  setRot('rightLowerArm', 0, -0.1, 0);
}

// ---------- motion data ----------
const clips = {};       // name -> {name, duration, channels: [{node, prop, interp}]}
let UPPER = new Set(), LOWER = new Set();   // normalized bone nodes
const UPPER_NAMES = ['spine', 'chest', 'upperChest', 'neck', 'head', 'jaw', 'leftShoulder', 'rightShoulder',
  'leftUpperArm', 'rightUpperArm', 'leftLowerArm', 'rightLowerArm', 'leftHand', 'rightHand'];
const LOWER_NAMES = ['hips', 'leftUpperLeg', 'leftLowerLeg', 'leftFoot', 'leftToes', 'rightUpperLeg', 'rightLowerLeg', 'rightFoot', 'rightToes'];
let FULL_BODY = new Set(['turn_around']);
const STATE_LOOPS = { idle: ['idle'], listen: ['listen'], think: ['think'], talk: ['talk', 'talk_2', 'talk_3'] };

async function loadMotions() {
  let names = [];
  try { names = JSON.parse(app.anims()); } catch (e) { names = []; }
  // the model side's manifest (bone groups, full-body clips), if it came with the model
  try {
    const m = await (await fetch('/pack/motion-manifest.json')).json();
    if (m.upperBodyBones) UPPER_NAMES.splice(0, UPPER_NAMES.length, ...m.upperBodyBones);
    if (m.lowerBodyBones) LOWER_NAMES.splice(0, LOWER_NAMES.length, ...m.lowerBodyBones);
    if (m.clips) FULL_BODY = new Set(m.clips.filter((c) => c.kind === 'fullBodyReaction').map((c) => c.name));
  } catch (e) { /* defaults above */ }
  const fingers = ['Thumb', 'Index', 'Middle', 'Ring', 'Little'];
  for (const n of UPPER_NAMES) { const b = bone(n); if (b) UPPER.add(b); }
  for (const side of ['left', 'right']) for (const f of fingers) for (const p of ['Metacarpal', 'Proximal', 'Intermediate', 'Distal']) {
    const b = bone(side + f + p); if (b) UPPER.add(b);
  }
  for (const n of LOWER_NAMES) { const b = bone(n); if (b) LOWER.add(b); }
  if (!names.length) return;
  const loader = new GLTFLoader();
  loader.register((p) => new VRMAnimationLoaderPlugin(p));
  for (const file of names) {
    try {
      const g = await loader.loadAsync('/pack/' + file);
      const anim = g.userData.vrmAnimations && g.userData.vrmAnimations[0];
      if (!anim) continue;
      const clip = createVRMAnimationClip(anim, vrm);
      const channels = [];
      for (const tr of clip.tracks) {
        const dot = tr.name.lastIndexOf('.');
        const node = vrm.scene.getObjectByName(tr.name.slice(0, dot));
        const prop = tr.name.slice(dot + 1);
        if (node && (prop === 'quaternion' || prop === 'position')) channels.push({ node, prop, interp: tr.createInterpolant() });
      }
      const name = file.replace(/\.vrma$/, '');
      clips[name] = { name, duration: clip.duration, channels };
    } catch (e) {
      log({ kind: 'avatar_error', message: 'motion ' + file + ': ' + (e && e.message || e) });
    }
  }
}

const tmpQ = new THREE.Quaternion(), tmpV = new THREE.Vector3();

/** Writes a clip's pose at time t onto the bones, blended by w from what they hold; only, if given, these nodes. */
function writeClip(clip, t, w, only, skipHipsPosition) {
  if (!clip || w <= 0) return;
  const time = Math.max(0, Math.min(t, clip.duration));
  for (const c of clip.channels) {
    if (only && !only.has(c.node)) continue;
    const v = c.interp.evaluate(time);
    if (c.prop === 'quaternion') {
      tmpQ.set(v[0], v[1], v[2], v[3]).normalize();
      if (w >= 1) c.node.quaternion.copy(tmpQ); else c.node.quaternion.slerp(tmpQ, w);
    } else {
      if (skipHipsPosition) continue;
      tmpV.set(v[0], v[1], v[2]);
      if (w >= 1) c.node.position.copy(tmpV); else c.node.position.lerp(tmpV, w);
    }
  }
}

// ---------- base layer: state loops and think transitions ----------
const base = { logical: 'idle', cur: null, prev: null, fade: 0, fadeT: 0 };   // cur/prev: {clip, t, loop}

function startBase(name, loop, fade) {
  const clip = clips[name];
  if (!clip) return false;
  base.prev = base.cur;
  base.cur = { clip, t: 0, loop };
  base.fade = base.prev ? fade : 0;
  base.fadeT = 0;
  return true;
}

function loopFor(state) {
  const list = (STATE_LOOPS[state] || ['idle']).filter((n) => clips[n]);
  if (!list.length) return clips.idle ? 'idle' : null;
  return list[Math.floor(Math.random() * list.length)];
}

function goTo(state, fade) {
  if (state === 'think' && clips.think_enter && clips.think) {
    startBase('think_enter', false, fade);
    base.logical = 'think_enter';
  } else {
    const name = loopFor(state);
    if (name) startBase(name, true, fade);
    base.logical = state;
  }
}

function updateBase(dt, wanted) {
  for (const s of [base.cur, base.prev]) {
    if (!s) continue;
    s.t += dt;
    if (s.loop && s.t >= s.clip.duration) {
      s.t %= s.clip.duration;
      // talk: sometimes move on to another variant at the end of a cycle
      if (s === base.cur && base.logical === 'talk' && Math.random() < 0.35) {
        const next = loopFor('talk');
        if (next && next !== s.clip.name) startBase(next, true, 0.4);
      }
    }
  }
  if (base.prev) { base.fadeT += dt; if (base.fadeT >= base.fade) base.prev = null; }
  const done = base.cur && !base.cur.loop && base.cur.t >= base.cur.clip.duration;
  if (base.logical === 'think_enter') {
    if (done) { startBase('think', true, 0.15); base.logical = 'think'; }
  } else if (base.logical === 'think_exit') {
    if (done) goTo(wanted, 0.15);
  } else if (wanted !== base.logical || !base.cur) {
    if (base.logical === 'think' && clips.think_exit) {
      startBase('think_exit', false, 0.15);
      base.logical = 'think_exit';
    } else {
      goTo(wanted, 0.3);
    }
  }
}

function writeBase() {
  if (!base.cur) return false;
  if (base.prev) {
    writeClip(base.prev.clip, base.prev.t, 1, null);
    writeClip(base.cur.clip, base.cur.t, clamp01(base.fadeT / base.fade), null);
  } else {
    writeClip(base.cur.clip, base.cur.t, 1, null);
  }
  return true;
}

// ---------- reactions ----------
let reaction = null;    // upper body: {name, clip, t, outAt}
let full = null;        // whole body: {name, clip, t}
let pendingStartle = false;

function startReaction(name) {
  const clip = clips[name];
  if (!clip || !vrm) return false;
  if (FULL_BODY.has(name)) {
    if (full || flat) return false;   // a repeated request doesn't restart it; a flat picture can't turn around
    full = { name, clip, t: 0 };
  } else {
    if (full || (reaction && reaction.name === name)) return false;
    reaction = { name, clip, t: 0, outAt: null };
  }
  log({ kind: 'avatar_motion', name, seconds: +clip.duration.toFixed(2) });
  try { if (app.motionStarted) app.motionStarted(name, clip.duration); } catch (e) { /* ignore */ }
  return true;
}
window.playReaction = startReaction;

/** The user talked over her: let a turn-around finish, otherwise leave the current reaction in 0.2 s, then startle. */
window.avatarInterrupt = () => {
  if (full) { pendingStartle = true; return; }
  if (reaction && reaction.name !== 'startle') reaction.outAt = reaction.t;
  pendingStartle = true;
};

function reactionWeight(r) {
  let w = Math.min(1, r.t / 0.2, (r.clip.duration - r.t) / 0.3);
  if (r.outAt !== null) w = Math.min(w, 1 - (r.t - r.outAt) / 0.2);
  return clamp01(w);
}

function updateReactions(dt) {
  if (reaction) {
    reaction.t += dt;
    if (reaction.t >= reaction.clip.duration || (reaction.outAt !== null && reaction.t - reaction.outAt >= 0.2)) reaction = null;
  }
  if (full) {
    full.t += dt;
    if (full.t >= full.clip.duration) full = null;
  }
  if (pendingStartle && !full && !reaction) {
    pendingStartle = false;
    startReaction(clips.startle ? 'startle' : 'surprised_recoil');
  }
}

// ---------- face ----------
const EMOTIONS = ['jito', 'sigh', 'smug', 'sleepy', 'giggle', 'happy', 'angry', 'sad', 'relaxed', 'surprised', 'shy', 'pout'];
const MOUTH_K = 0.3;   // <emo>_mouth while speaking (expression-composition.json, initial value)
const EMOJI = { '😒': 'jito', '🙄': 'sigh', '😏': 'smug', '😪': 'sleepy', '🤭': 'giggle' };
const REACTION_BY_EMOTION = { sleepy: 'sleepy_nod', giggle: 'giggle_hide', sigh: 'sigh_drop', surprised: 'surprised_recoil' };
const ACTIONS = { 'うなずく': 'nod', '首を振る': 'shake', '首をかしげる': 'tilt', '肩をすくめる': 'shrug', 'ため息': 'sigh_drop',
  'ツッコミ': 'tsukkomi', 'くすっと笑う': 'giggle_hide', '目をそらす': 'look_away', '腕を組む': 'cross_arms', '手を振る': 'wave',
  'あくび': 'yawn', 'びくっ': 'surprised_recoil', '身を乗り出す': 'lean_in', '後ろを向く': 'turn_around', 'うとうと': 'sleepy_nod' };
// timed eye closing during giggle_hide (reaction-face-cues.json); yawn uses the same timing
const BLINK_CUES = { giggle_hide: { closeStart: 1.92, closedAt: 2.08, openStart: 3.38, openedAt: 3.62, peak: 1 } };
BLINK_CUES.yawn = BLINK_CUES.giggle_hide;

const face = { emotion: null, weights: {}, idleSince: 0, nextBlink: 2, blinkT: -1, vowels: [], cueAt: 0,
  mouth: { aa: 0, ih: 0, ou: 0, ee: 0, oh: 0 }, gaze: 0, voice: 0 };

const smoothstep = (a, b, x) => { const u = clamp01((x - a) / (b - a)); return u * u * (3 - 2 * u); };
function blinkCue(cue, t) {
  return cue ? cue.peak * smoothstep(cue.closeStart, cue.closedAt, t) * (1 - smoothstep(cue.openStart, cue.openedAt, t)) : 0;
}

// kana -> vowel expression; small ゃゅょ change the vowel of the mora before; ん/っ/ー/punctuation close or hold
const VOWEL = {};
const ROWS = ['あいうえお', 'かきくけこ', 'がぎぐげご', 'さしすせそ', 'ざじずぜぞ', 'たちつてと', 'だぢづでど', 'なにぬねの',
  'はひふへほ', 'ばびぶべぼ', 'ぱぴぷぺぽ', 'まみむめも', 'らりるれろ', 'ぁぃぅぇぉ'];
const V5 = ['aa', 'ih', 'ou', 'ee', 'oh'];
for (const row of ROWS) for (let i = 0; i < 5; i++) VOWEL[row[i]] = V5[i];
Object.assign(VOWEL, { 'や': 'aa', 'ゆ': 'ou', 'よ': 'oh', 'わ': 'aa', 'を': 'oh', 'ゔ': 'ou' });
const SMALL_Y = { 'ゃ': 'aa', 'ゅ': 'ou', 'ょ': 'oh' };

function vowelsOf(text) {
  const out = [];
  let kanji = 0;
  for (const ch0 of text) {
    let ch = ch0;
    const c = ch.codePointAt(0);
    if (c >= 0x30a1 && c <= 0x30f6) ch = String.fromCodePoint(c - 0x60);   // katakana -> hiragana
    if (SMALL_Y[ch] && out.length) { out[out.length - 1] = SMALL_Y[ch]; continue; }
    if (VOWEL[ch]) out.push(VOWEL[ch]);
    else if (ch === 'ー') out.push(out.length ? out[out.length - 1] : 'aa');
    else if (ch === 'ん' || ch === 'っ') out.push(null);
    else if (c >= 0x4e00 && c <= 0x9fff) { out.push(kanji % 2 ? 'oh' : 'aa', kanji % 3 ? 'ih' : 'ou'); kanji++; }   // unknown reading: a guess
    else if ('、。！？…!?,. '.includes(ch)) out.push(null, null);
  }
  return out;
}

/** A sentence starts playing: its emotion (emoji), action tags and text. */
window.avatarCue = (cue) => {
  if (!vrm) return;
  face.vowels = vowelsOf(cue.text || '');
  face.cueAt = clock.elapsedTime;
  const emo = cue.emotion ? (EMOJI[cue.emotion] || cue.emotion) : null;
  let acted = false;
  for (const a of cue.actions || []) {
    const name = ACTIONS[a] || (clips[a] ? a : null);
    if (name && startReaction(name)) acted = true;
  }
  if (emo && EMOTIONS.includes(emo)) {
    const entering = emo !== face.emotion;
    face.emotion = emo;
    if (entering && !acted && REACTION_BY_EMOTION[emo]) startReaction(REACTION_BY_EMOTION[emo]);
  } else if (cue.first) {
    face.emotion = null;   // a new reply without an emotion starts from the neutral face
  }
};

const MORA_S = 0.13;   // seconds per mora at the default tempo (1.2)

function updateFace(dt, state, level) {
  const speaking = state === 'talk';
  // emotion weights: towards 1 for the current emotion, 0 for the others (0.2 s)
  if (state === 'idle' && face.emotion) {
    face.idleSince += dt;
    if (face.idleSince > 10) face.emotion = null;   // back to the neutral face after a while
  } else {
    face.idleSince = 0;
  }
  for (const e of EMOTIONS) {
    const target = e === face.emotion ? 1 : 0;
    const w = face.weights[e] || 0;
    face.weights[e] = w + Math.sign(target - w) * Math.min(Math.abs(target - w), dt / 0.2);
  }
  // natural blink, combined with the reaction's timed cue
  if (clock.elapsedTime > face.nextBlink && face.blinkT < 0) face.blinkT = 0;
  let nat = 0;
  if (face.blinkT >= 0) {
    face.blinkT += dt;
    nat = face.blinkT < 0.07 ? face.blinkT / 0.07 : face.blinkT < 0.15 ? 1 - (face.blinkT - 0.07) / 0.08 : 0;
    if (face.blinkT >= 0.15) { face.blinkT = -1; face.nextBlink = clock.elapsedTime + 2 + Math.random() * 4; }
  }
  let cue = 0, yawnOpen = 0;
  if (reaction && BLINK_CUES[reaction.name]) {
    cue = blinkCue(BLINK_CUES[reaction.name], reaction.t) * reactionWeight(reaction);
    if (reaction.name === 'yawn') yawnOpen = 0.5 * cue;
  }
  const b = 1 - (1 - nat) * (1 - cue);
  let yawnSleepy = reaction && reaction.name === 'yawn' ? reactionWeight(reaction) : 0;

  for (const e of EMOTIONS) {
    let w = face.weights[e] || 0;
    if (e === 'sleepy') w = Math.max(w, yawnSleepy);
    setExpr(e, w);
    setExpr(e + '_lids', w * (1 - b));
    setExpr(e + '_mouth', w * (speaking ? MOUTH_K : 1));
  }
  setExpr('blink', b);

  // mouth: openness from the voice, vowel from the sentence's kana at the estimated position
  face.voice += (clamp01(level / 0.08) - face.voice) * (level > 0 ? 0.6 : 0.3);
  let vowel = 'aa';
  if (face.vowels.length) {
    const i = Math.floor((clock.elapsedTime - face.cueAt) / MORA_S);
    vowel = i < face.vowels.length ? face.vowels[i] : 'aa';
  }
  const open = speaking ? face.voice : 0;
  const k = Math.min(1, dt / 0.06);   // 60 ms between vowels
  let sum = 0;
  for (const v of V5) {
    const target = v === vowel && vowel ? open : 0;
    face.mouth[v] += (target - face.mouth[v]) * (speaking ? k : Math.min(1, dt / 0.1));
    sum += face.mouth[v];
  }
  const scale = sum > 1 ? 1 / sum : 1;   // the vowels never add up to more than 1
  for (const v of V5) setExpr(v, face.mouth[v] * scale + (v === 'aa' ? yawnOpen : 0));

  // gaze: up and to the side while thinking
  face.gaze += ((state === 'think' ? 1 : 0) - face.gaze) * Math.min(1, dt / 0.3);
  setExpr('lookUp', 0.35 * face.gaze);
  setExpr('lookLeft', 0.2 * face.gaze);
}

// ---------- life layer: breathing and a slight sway on top of every motion ----------
// The clips other than idle hold still between their gestures, so this always adds a little: the chest
// and back breathing (3.6 s), the shoulders rising with it, the upper body swaying side to side (7 s)
// and the head drifting on slower, unrelated periods. Lighter on idle (it breathes already) and
// during a turn-around. Legs and hips are untouched, so the feet stay where they are.
const life = { t: 0, w: 0 };
const qOff = new THREE.Quaternion(), eOff = new THREE.Euler();

function addRot(name, x, y, z) {
  const b = bone(name);
  if (!b) return;
  qOff.setFromEuler(eOff.set(x, y, z));
  b.quaternion.multiply(qOff);   // in the bone's own space, after the clip's rotation
}

function applyLife(dt) {
  life.t += dt;
  const target = full ? 0.4 : (base.cur && base.cur.clip.name === 'idle' && !reaction) ? 0.35 : 1;
  life.w += (target - life.w) * Math.min(1, dt / 0.5);
  const w = life.w, t = life.t;
  const breath = Math.sin(2 * Math.PI * t / 3.6);
  const sway = Math.sin(2 * Math.PI * t / 7.0);
  const drift = Math.sin(2 * Math.PI * t / 9.3 + 1.1), tilt = Math.sin(2 * Math.PI * t / 5.7 + 0.4);
  addRot('spine', 0.008 * breath * w, 0.012 * sway * w, 0.010 * sway * w);
  addRot('chest', 0.014 * breath * w, 0, 0.006 * sway * w);
  addRot('upperChest', 0.008 * breath * w, 0, 0);
  addRot('leftShoulder', 0, 0, 0.010 * breath * w);
  addRot('rightShoulder', 0, 0, -0.010 * breath * w);
  addRot('neck', -0.006 * breath * w, 0.010 * drift * w, 0);
  addRot('head', 0.008 * drift * w, 0.018 * drift * w, 0.010 * tilt * w);
}

// ---------- procedural fallback (models without VRMA loops) ----------
let pt = 0, pvoice = 0, ptilt = 0;
function animateProcedural(dt, state, level) {
  pt += dt;
  pvoice += (Math.min(1, level / 0.1) - pvoice) * (level > pvoice * 0.1 ? 0.5 : 0.12);
  const breath = Math.sin(pt * 2 * Math.PI / 3.6), sway = Math.sin(pt * 2 * Math.PI / 7.0);
  const targetTilt = state === 'listen' || state === 'recognize' ? 0.12 : state === 'think' ? -0.08 : 0;
  ptilt += (targetTilt - ptilt) * Math.min(1, dt * 4);
  poseRest();
  setRot('spine', 0.015 * breath, 0.02 * sway, 0);
  setRot('chest', 0.02 * breath, 0, 0.01 * sway);
  const talkNod = state === 'talk' ? 0.05 * pvoice * Math.sin(pt * 9) : 0;
  setRot('neck', 0.01 * breath + talkNod * 0.5, 0.03 * sway, 0);
  setRot('head', (state === 'think' ? -0.08 : 0) + talkNod + 0.02 * pvoice, 0.04 * sway + (state === 'think' ? 0.15 : 0), ptilt);
  setRot('leftUpperArm', 0, 0, -1.15 - 0.015 * breath);
  setRot('rightUpperArm', 0, 0, 1.15 + 0.015 * breath);
}

const STATE_OF = { loading: 'idle', idle: 'idle', listen: 'listen', recognize: 'listen', think: 'think', talk: 'talk' };

/** One animation step: base layer, reactions, face; then vrm.update (humanoid, expressions, springs). */
function step(dt, appState, level) {
  const state = STATE_OF[appState] || 'idle';
  vrm.humanoid.resetNormalizedPose();   // every bone from rest, so nothing is left over from a motion
  updateBase(dt, state);
  if (!writeBase()) animateProcedural(dt, appState, level);
  updateReactions(dt);
  if (reaction) writeClip(reaction.clip, reaction.t, reactionWeight(reaction), UPPER);   // upper body only
  if (full) writeClip(full.clip, full.t, clamp01(Math.min(full.t / 0.3, (full.clip.duration - full.t) / 0.3)), null);
  if (base.cur) applyLife(dt);
  updateFace(dt, state, level);
  if (flat) flat.update(flatLocal, bone('hips').position, dt);   // the 3D model itself is not drawn
  else vrm.update(dt);
}

renderer.domElement.addEventListener('pointerup', () => startReaction('turn_around'));

// for tests (adb / PC browser): what is playing
window.avatarDebug = () => ({
  base: base.logical, baseClip: base.cur && base.cur.clip.name, prev: base.prev && base.prev.clip.name,
  reaction: reaction && { name: reaction.name, t: +reaction.t.toFixed(2) },
  full: full && { name: full.name, t: +full.t.toFixed(2) },
  emotion: face.emotion, clips: Object.keys(clips).length,
  hips: bone('hips') && bone('hips').quaternion.toArray().map((v) => +v.toFixed(3)),
  chest: bone('chest') && bone('chest').quaternion.toArray().map((v) => +v.toFixed(4)),
  life: +life.w.toFixed(2),
});

// ---- render loop, capped at 60 fps; only frames actually drawn are counted ----
const clock = new THREE.Clock();
let lastDraw = 0, nextDue = 0;
const stats = { since: performance.now(), frames: 0, intervals: [], talk: 0 };

function report(now) {
  const iv = stats.intervals.slice().sort((a, b) => a - b);
  if (iv.length > 10) {
    const q = (p) => +iv[Math.min(iv.length - 1, Math.floor(p * iv.length))].toFixed(1);
    const sec = (now - stats.since) / 1000;
    log({ kind: 'avatar_perf', seconds: +sec.toFixed(1), fps: +(stats.frames / sec).toFixed(1),
          p50_ms: q(0.5), p99_ms: q(0.99), max_ms: +iv[iv.length - 1].toFixed(1),
          talking_share: +(stats.talk / stats.frames).toFixed(2),
          pixelRatio: renderer.getPixelRatio(), calls: renderer.info.render.calls, tris: renderer.info.render.triangles });
  }
  stats.since = now; stats.frames = 0; stats.intervals = []; stats.talk = 0;
}

function loop(now) {
  requestAnimationFrame(loop);
  if (document.hidden) return;
  // Draw on the first screen refresh at or after the next 60 fps slot (keeping the slot phase), so a
  // 90 or 120 Hz screen still averages 60. Skipped refreshes are not counted as frames.
  if (now < nextDue - 4) return;
  nextDue = Math.max(nextDue + 1000 / FPS_CAP, now);
  const dtMs = lastDraw ? now - lastDraw : 16.7;
  lastDraw = now;
  const dt = Math.min(clock.getDelta(), 1 / 20);
  if (vrm) {
    let state = 'idle', level = 0;
    try { state = app.state(); level = app.level(); } catch (e) { /* ignore */ }
    step(dt, state, level);
    if (state === 'talk') stats.talk++;
  }
  draw();
  stats.frames++;
  stats.intervals.push(dtMs);
  if (now - stats.since > 10000) report(now);
}

// for tests: advance the animation by hand (a hidden browser tab doesn't run requestAnimationFrame)
window.avatarStep = (seconds, state = 'idle', level = 0, fps = 30) => {
  for (let i = 0; i < Math.round(seconds * fps); i++) step(1 / fps, state, level);
  draw();
  return window.avatarDebug();
};

load().catch((e) => log({ kind: 'avatar_error', message: String(e && e.message || e) }));
requestAnimationFrame(loop);
