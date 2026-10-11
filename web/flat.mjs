// The character as a flat picture (2D) driven by the same humanoid pose as the 3D model.
//
// The picture comes as layers (tools/flat_bake.py: See-through parts, "visible = original picture") plus a
// rig.json with joint positions in picture pixels. Each layer is a grid skinned to a 2D skeleton. Every frame
// the normalized humanoid rotations (what avatar.mjs writes from the VRMA clips, reactions and the breathing
// layer) are composed into world rotations, taken relative to a reference pose that matches the picture
// (idle at t=0), applied to each 2D bone's drawn direction and projected onto the picture plane: the in-plane
// angle turns the 2D bone. A segment that points at the viewer has no reliable angle and follows its parent.
// Head yaw/pitch slides the face parts a little (and the back hair the other way).
//
// Face, without extra pictures: each eye (eyewhite/irides/eyelash-l|r layers) sits on its own bone that
// squashes from the lower lid for blinks and half-closed lids (the squashed upper lashes read as a closed
// eye); the mouth opens as a small dark shape at joints.mouth, sized by vowel and voice; shy/happy add a
// blush. The values come from avatar.mjs (face.blink, face.vowelOut, face.weights).
//
// rig.json: {"layers": "layers_flat.json", "joints": {"pelvis": [x, y], ..., "mouth": [x, y]}, "layerBones": [[regex, [bones]], ...],
//            "sigma": px, "cell": px, "facePx": px}
// layers_flat.json: [{"name", "filename", "left", "top"}, ...] bottom to top.
import * as THREE from 'three';

// 2D bone, parent, from joint, to joint, humanoid bone whose world rotation drives it
const BONES = [
  ['hips', null, 'pelvis', 'waist', 'hips'],
  ['spine', 'hips', 'waist', 'chest', 'spine'],
  ['chest', 'spine', 'chest', 'neck', 'upperChest'],
  ['neck', 'chest', 'neck', 'head', 'neck'],
  ['head', 'neck', 'head', 'headTop', 'head'],
];
for (const [s, side] of [['l', 'left'], ['r', 'right']]) {
  BONES.push([s + 'UpperArm', 'chest', s + 'Shoulder', s + 'Elbow', side + 'UpperArm'],
    [s + 'LowerArm', s + 'UpperArm', s + 'Elbow', s + 'Wrist', side + 'LowerArm'],
    [s + 'Hand', s + 'LowerArm', s + 'Wrist', s + 'HandTip', side + 'Hand'],
    [s + 'UpperLeg', 'hips', s + 'Hip', s + 'Knee', side + 'UpperLeg'],
    [s + 'LowerLeg', s + 'UpperLeg', s + 'Knee', s + 'Ankle', side + 'LowerLeg'],
    [s + 'Foot', s + 'LowerLeg', s + 'Ankle', s + 'Toe', side + 'Foot']);
}
BONES.push(['tail', 'hips', 'tailBase', 'tailTip', null]);

// humanoid parent chain (normalized bones: rest = identity, so world = product of locals)
const CHAIN = { hips: null, spine: 'hips', chest: 'spine', upperChest: 'chest', neck: 'upperChest', head: 'neck' };
for (const side of ['left', 'right']) Object.assign(CHAIN, {
  [side + 'Shoulder']: 'upperChest', [side + 'UpperArm']: side + 'Shoulder', [side + 'LowerArm']: side + 'UpperArm',
  [side + 'Hand']: side + 'LowerArm', [side + 'UpperLeg']: 'hips', [side + 'LowerLeg']: side + 'UpperLeg', [side + 'Foot']: side + 'LowerLeg',
});
export const HUMANOID_BONES = Object.keys(CHAIN);

const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));
const P = (p) => new THREE.Vector3(p[0], -p[1], 0);   // picture px (y down) -> world (y up)

function segDist(p, a, b) {
  const ab = b.clone().sub(a), t = THREE.MathUtils.clamp(p.clone().sub(a).dot(ab) / ab.lengthSq(), 0, 1);
  return p.distanceTo(a.clone().addScaledVector(ab, t));
}

/**
 * Loads the rig and layers. local(name) gives a humanoid bone's normalized local quaternion (or null),
 * hipsPosition() its normalized position, both read each update.
 */
export async function createFlat(rigUrl) {
  const rig = await (await fetch(rigUrl)).json();
  const base = rigUrl.slice(0, rigUrl.lastIndexOf('/') + 1);
  const layers = await (await fetch(base + rig.layers)).json();
  const J = rig.joints;
  const loader = new THREE.TextureLoader();
  const textures = await Promise.all(layers.map((L) => loader.loadAsync(base + encodeURIComponent(L.filename))));

  const bones = {}, info = {};
  const root = new THREE.Group();
  for (const [name, parent, from, to, hum] of BONES) {
    if (!J[from] || !J[to]) continue;
    const b = new THREE.Bone();
    b.name = name;
    const p = P(J[from]);
    if (parent) { b.position.copy(p).sub(P(J[BONES.find((x) => x[0] === parent)[2]])); bones[parent].add(b); }
    else { b.position.copy(p); root.add(b); }
    bones[name] = b;
    const d0 = P(J[to]).sub(p);
    info[name] = { parent, hum, from: p, to: P(J[to]), d0, rest: Math.atan2(d0.y, d0.x), angle: 0 };
  }
  for (const n of ['face', 'back']) { const b = new THREE.Bone(); b.name = n; bones.head.add(b); bones[n] = b; }
  // eyes: a bone at each lower lid, under the face bone
  const eyes = {};
  layers.forEach((L, i) => {
    const m = L.name.match(/^eyewhite-([lr])$/);
    if (!m) return;
    const w = textures[i].image.width, h = textures[i].image.height;
    eyes[m[1]] = { x: L.left + w / 2, y: L.top + h * 0.85, w, h };
  });
  // the eyes' tilt in the picture (a tilted head): lids close across it, not straight down
  const tilt = eyes.l && eyes.r ? Math.atan2(-(eyes.l.y - eyes.r.y), eyes.l.x - eyes.r.x) : 0;
  for (const [side, e] of Object.entries(eyes)) {
    const b = new THREE.Bone();
    b.name = 'eye' + side;
    b.position.copy(P([e.x, e.y])).sub(P(J.head));
    b.rotation.z = tilt;
    bones.face.add(b);
    bones['eye' + side] = b;
  }
  const boneList = Object.values(bones);
  root.updateMatrixWorld(true);
  const skeleton = new THREE.Skeleton(boneList);

  const scene = new THREE.Scene();
  scene.add(root);
  const SIGMA = rig.sigma || 30, CELL = rig.cell || 10;
  const layerBones = (name) => {
    name = name.replace(/__back$/, '');   // a part's cut-away back piece moves with the part
    const eye = name.match(/^(eyewhite|irides|eyelash)-([lr])$/);
    if (eye && bones['eye' + eye[2]]) return ['eye' + eye[2]];
    for (const [re, bs] of rig.layerBones) if (new RegExp(re).test(name)) return bs.filter((n) => bones[n]);
    return ['hips', 'spine', 'chest'].filter((n) => bones[n]);
  };
  let top = Infinity;
  layers.forEach((L, order) => {
    const tex = textures[order];
    tex.colorSpace = THREE.SRGBColorSpace;
    const w = tex.image.width, h = tex.image.height;
    top = Math.min(top, L.top);
    const g = new THREE.PlaneGeometry(w, h, Math.max(1, Math.round(w / CELL)), Math.max(1, Math.round(h / CELL)));
    g.translate(L.left + w / 2, -(L.top + h / 2), 0);
    const allowed = layerBones(L.name);
    const pos = g.attributes.position, n = pos.count;
    const si = new Uint16Array(n * 4), sw = new Float32Array(n * 4);
    const v = new THREE.Vector3();
    for (let i = 0; i < n; i++) {
      v.fromBufferAttribute(pos, i);
      // weights fall off with the distance to each allowed bone segment (face/back: rigid on their bone)
      const ds = allowed.map((bn) => [bn, info[bn] ? segDist(v, info[bn].from, info[bn].to) : 0]);
      const dmin = Math.min(...ds.map((x) => x[1]));
      let ws = ds.map(([bn, d]) => [boneList.indexOf(bones[bn]), Math.exp(-(d * d - dmin * dmin) / (SIGMA * SIGMA))]);
      ws.sort((a, b) => b[1] - a[1]);
      ws = ws.slice(0, 4);
      const sum = ws.reduce((s, x) => s + x[1], 0);
      ws.forEach(([bi, wt], k) => { si[i * 4 + k] = bi; sw[i * 4 + k] = wt / sum; });
    }
    g.setAttribute('skinIndex', new THREE.Uint16BufferAttribute(si, 4));
    g.setAttribute('skinWeight', new THREE.Float32BufferAttribute(sw, 4));
    const m = new THREE.SkinnedMesh(g, new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthTest: false, depthWrite: false }));
    m.renderOrder = order;
    m.frustumCulled = false;
    m.name = L.name;
    m.bind(skeleton, new THREE.Matrix4());
    scene.add(m);
  });

  // mouth: a small dark opening (and a lighter inside) on the face bone; cheeks: a soft blush
  let mouth = null;
  if (J.mouth) {
    mouth = new THREE.Group();
    mouth.position.copy(P(J.mouth)).sub(P(J.head));
    const outer = new THREE.Mesh(new THREE.CircleGeometry(1, 24), new THREE.MeshBasicMaterial({ color: 0x4a1c22, transparent: true, depthTest: false }));
    const inner = new THREE.Mesh(new THREE.CircleGeometry(1, 24), new THREE.MeshBasicMaterial({ color: 0xa8505c, transparent: true, depthTest: false }));
    inner.position.set(0, -0.45, 0); inner.scale.set(0.6, 0.45, 1);
    outer.renderOrder = 10000; inner.renderOrder = 10001;
    mouth.add(outer); outer.add(inner);
    mouth.userData.outer = outer;
    mouth.visible = false;
    bones.face.add(mouth);
  }
  const blushTex = (() => {
    const c = document.createElement('canvas'); c.width = c.height = 64;
    const g = c.getContext('2d'), gr = g.createRadialGradient(32, 32, 0, 32, 32, 32);
    gr.addColorStop(0, 'rgba(255,105,125,0.9)'); gr.addColorStop(1, 'rgba(255,105,125,0)');
    g.fillStyle = gr; g.fillRect(0, 0, 64, 64);
    const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
  })();
  const blush = Object.entries(eyes).map(([side, e]) => {
    const m = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({ map: blushTex, transparent: true, depthTest: false, opacity: 0 }));
    const out = side === 'l' ? 1 : -1;   // character's left = picture right
    m.position.copy(P([e.x - out * e.w * 0.12, e.y + e.h * 0.75])).sub(P(J.head));
    m.rotation.z = tilt;
    m.scale.set(e.w * 1.1, e.w * 0.5, 1);
    m.renderOrder = 9990;
    bones.face.add(m);
    return m;
  });
  // closed eyes: a lash line along the lower lid, faded in as the lids close
  const lashTex = (mirror) => {
    const c = document.createElement('canvas'); c.width = 128; c.height = 32;
    const g = c.getContext('2d');
    if (mirror) { g.translate(128, 0); g.scale(-1, 1); }
    g.strokeStyle = '#3a2228'; g.lineCap = 'round'; g.lineWidth = 3.5;
    g.beginPath(); g.moveTo(8, 9); g.quadraticCurveTo(62, 25, 118, 11); g.stroke();
    g.lineWidth = 2.5; g.beginPath(); g.moveTo(112, 12); g.lineTo(124, 6); g.stroke();   // outer corner
    const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
  };
  const lashes = Object.entries(eyes).map(([side, e]) => {
    const m = new THREE.Mesh(new THREE.PlaneGeometry(1, 1),
      new THREE.MeshBasicMaterial({ map: lashTex(side === 'r'), transparent: true, depthTest: false, opacity: 0 }));
    m.position.copy(P([e.x, e.y - e.h * 0.12])).sub(P(J.head));
    m.rotation.z = tilt;
    m.scale.set(e.w * 1.1, e.w * 1.1 / 4, 1);
    m.renderOrder = 9995;
    bones.face.add(m);
    return m;
  });

  // framing: top of the picture to the knees, like the 3D view (ears to knees)
  const knees = ['lKnee', 'rKnee'].filter((k) => J[k]).map((k) => J[k][1]);
  const span = { top, bottom: knees.length ? knees.reduce((a, b) => a + b) / knees.length : top + 800,
    x: J.pelvis ? J.pelvis[0] : 640 };
  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, -10, 10);
  function frame(aspect, viewTop, viewBottom) {
    const H = (span.bottom - span.top) / (viewBottom - viewTop);
    const yTop = -(span.top - viewTop * H);
    camera.top = yTop; camera.bottom = yTop - H;
    camera.left = span.x - H * aspect / 2; camera.right = span.x + H * aspect / 2;
    camera.updateProjectionMatrix();
  }

  const world = (local) => {
    const out = {};
    for (const name of HUMANOID_BONES) {
      const q = local(name) || new THREE.Quaternion();
      const p = CHAIN[name];
      out[name] = p ? out[p].clone().multiply(q) : q.clone();
    }
    return out;
  };
  let ref = null, refHips = new THREE.Vector3();
  const pxPerM = { v: 0 };
  const FACE_PX = rig.facePx ?? 10;
  let t = 0;
  // emotion -> how far the lids come down, extra eye opening, blush
  const LIDS = { jito: 0.45, sigh: 0.35, smug: 0.3, sleepy: 0.6, giggle: 0.5, happy: 0.3, angry: 0.25, sad: 0.2, relaxed: 0.3, shy: 0.2, pout: 0.15 };
  const BLUSH = { shy: 1, happy: 0.4, giggle: 0.5, pout: 0.3 };
  // mouth shape per vowel: half width, half height at full opening (picture px)
  const VOWELS = { aa: [4.5, 3.6], ih: [5, 1.6], ou: [2.6, 2.4], ee: [5, 2.2], oh: [3.5, 3.2] };
  function updateFace(face) {
    if (!face) return;
    const W = face.weights || {};
    let lids = 0, wide = 0, bl = 0;
    for (const [e, w] of Object.entries(W)) {
      lids = Math.max(lids, (LIDS[e] || 0) * w);
      bl = Math.max(bl, (BLUSH[e] || 0) * w);
    }
    wide = W.surprised || 0;
    const close = 1 - (1 - (face.blink || 0)) * (1 - lids);
    for (const side of ['l', 'r']) {
      const b = bones['eye' + side];
      if (!b) continue;
      b.scale.set(1 + 0.04 * wide, Math.max(0.06, (1 - close) * (1 + 0.12 * wide)), 1);
    }
    for (const m of blush) m.material.opacity = 0.55 * bl;
    for (const m of lashes) m.material.opacity = THREE.MathUtils.smoothstep(close, 0.55, 0.85);
    if (mouth) {
      let hw = 0, hh = 0, sum = 0;
      for (const [v, [w, h]] of Object.entries(VOWELS)) {
        const k = (face.vowelOut && face.vowelOut[v]) || 0;
        hw += w * k; hh += h * k; sum += k;
      }
      if (W.surprised) { hw = Math.max(hw, 2.2 * W.surprised); hh = Math.max(hh, 2.2 * W.surprised); sum = Math.max(sum, W.surprised); }
      mouth.visible = sum > 0.05;
      if (mouth.visible) {
        const k = Math.min(1, sum);
        mouth.userData.outer.scale.set(Math.max(2.5, hw / Math.max(sum, 1e-3)), Math.max(0.6, hh / Math.max(sum, 1e-3) * k), 1);
      }
    }
  }

  return {
    scene, camera, frame,
    /** The pose that matches the picture (idle at t=0); hipsRestY: the model's hips height in metres. */
    setReference(local, hips, hipsRestY) {
      ref = world(local);
      refHips = hips.clone();
      const legPx = (J.lAnkle && J.pelvis) ? J.lAnkle[1] - J.pelvis[1] : 500;
      pxPerM.v = legPx / (hipsRestY || 0.9);
    },
    update(local, hips, dt, face) {
      if (!ref) return;
      updateFace(face);
      t += dt;
      const wp = world(local);
      for (const [name, f] of Object.entries(info)) {
        let ang = 0;
        const pa = f.parent ? info[f.parent].angle : 0;
        if (f.hum) {
          const delta = wp[f.hum].clone().multiply(ref[f.hum].clone().invert());
          const d = f.d0.clone().applyQuaternion(delta);
          ang = wrap(Math.atan2(d.y, d.x) - f.rest);
          const k = THREE.MathUtils.smoothstep(Math.hypot(d.x, d.y) / f.d0.length(), 0.35, 0.8);
          ang = pa + wrap(ang - pa) * k;
        } else if (name === 'tail') {
          ang = pa + 0.12 * Math.sin(t * 2.1);
        }
        f.angle = ang;
      }
      for (const [name, f] of Object.entries(info)) bones[name].rotation.z = f.angle - (f.parent ? info[f.parent].angle : 0);
      const hd = wp.head.clone().multiply(ref.head.clone().invert());
      const fwd = new THREE.Vector3(0, 0, 1).applyQuaternion(hd);
      bones.face.position.set(fwd.x * FACE_PX, fwd.y * FACE_PX, 0);
      bones.back.position.set(-fwd.x * FACE_PX * 0.5, -fwd.y * FACE_PX * 0.5, 0);
      const dh = hips.clone().sub(refHips);
      bones.hips.position.set(info.hips.from.x + dh.x * pxPerM.v, info.hips.from.y + dh.y * pxPerM.v, 0);
    },
    angles: () => Object.fromEntries(Object.entries(info).map(([k, f]) => [k, +(f.angle * 180 / Math.PI).toFixed(1)])),
  };
}
