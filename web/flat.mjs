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
// rig.json: {"layers": "layers_flat.json", "joints": {"pelvis": [x, y], ...}, "layerBones": [[regex, [bones]], ...],
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
  const boneList = Object.values(bones);
  root.updateMatrixWorld(true);
  const skeleton = new THREE.Skeleton(boneList);

  const scene = new THREE.Scene();
  scene.add(root);
  const loader = new THREE.TextureLoader();
  const SIGMA = rig.sigma || 30, CELL = rig.cell || 10;
  const layerBones = (name) => {
    name = name.replace(/__back$/, '');   // a part's cut-away back piece moves with the part
    for (const [re, bs] of rig.layerBones) if (new RegExp(re).test(name)) return bs.filter((n) => bones[n]);
    return ['hips', 'spine', 'chest'].filter((n) => bones[n]);
  };
  let top = Infinity;
  const textures = await Promise.all(layers.map((L) => loader.loadAsync(base + encodeURIComponent(L.filename))));
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

  return {
    scene, camera, frame,
    /** The pose that matches the picture (idle at t=0); hipsRestY: the model's hips height in metres. */
    setReference(local, hips, hipsRestY) {
      ref = world(local);
      refHips = hips.clone();
      const legPx = (J.lAnkle && J.pelvis) ? J.lAnkle[1] - J.pelvis[1] : 500;
      pxPerM.v = legPx / (hipsRestY || 0.9);
    },
    update(local, hips, dt) {
      if (!ref) return;
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
