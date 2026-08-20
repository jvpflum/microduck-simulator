// Microduck RL playground: the REAL trained policies, not a procedural waddle.
//
// Physics runs in MuJoCo compiled to WebAssembly (the official
// @mujoco/mujoco bindings), stepping the same MJCF the policies were
// trained on (apirrone/mjlab_microduck). The controller is one of the
// exported ONNX checkpoints from apirrone/microduck_runtime, executed with
// onnxruntime-web at 50 Hz (timestep 0.005 s, decimation 4) - exactly the
// loop from mjlab_microduck/scripts/infer_policy.py.
//
// Obs layout (61D, "new-cmd-obs" flavor, from the ONNX metadata):
//   [base_ang_vel(3), projected_gravity(3), joint_pos(14), joint_vel(14),
//    last_action(14), command(13)]
// command = [vx, vy, wz, head_pose(4), body_pose(6)]; for the sitstand
// policy, command[0] is the posture flag (1 = sit, 0 = stand).
// Action (14) = joint position targets relative to the default pose.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import loadMujoco from "https://cdn.jsdelivr.net/npm/@mujoco/mujoco@3.11.0/mujoco.js";
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/ort.min.mjs";

// Private HF Space auth: the hub iframe URL carries a ?__sign JWT, but
// subresource requests normally rely on a *.static.hf.space cookie that
// browsers often block inside the iframe (third-party cookie blocking),
// which 401s every same-origin fetch. Appending the JWT to each request
// authenticates them regardless of cookie policy. No-op locally.
const HF_SIGN = new URLSearchParams(location.search).get("__sign");
const signed = (url) =>
  HF_SIGN ? `${url}${url.includes("?") ? "&" : "?"}__sign=${encodeURIComponent(HF_SIGN)}` : url;
window.__hfSigned = signed; // duck.js uses it for kinematics + STL requests

// Local modules are imported dynamically through signed() for the same
// reason: a static import of ./duck.js would 401 without the cookie.
const { buildRig, loadKinematics, setJoint, setJawOpen, MODEL_DIR, MESH_VERSION } =
  await import(signed("./duck.js"));
const { VARIANTS, VARIANT_NAMES, materialHookFor, randomVariantName, applyVariant, specToHex } =
  await import(signed("./variants.js"));

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/";
ort.env.wasm.numThreads = 1; // static hosting sends no COOP/COEP headers

const POLICY_DIR = "./policies";
const POLICIES = {
  walk: `${POLICY_DIR}/BEST_alpha_walking.onnx`,
  sitstand: `${POLICY_DIR}/BEST_alpha_sitstand.onnx`,
  roulade: `${POLICY_DIR}/roulade.onnx`,
};

// From the ONNX metadata (identical for all alpha policies) and the STAND
// keyframe in mjlab's scene_walk.xml. Order matches the actuators in
// the MJCF.
const JOINT_NAMES = [
  "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
  "neck_pitch", "head_pitch", "head_yaw", "head_roll",
  "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
];
const DEFAULT_POSE = new Float32Array([
  0, -0.08726646259971647, -0.457924, -0.004940, 0.452984,
  0.3490658503988659, 0.3490658503988659, 0, 0,
  0, 0.08726646259971647, 0.457924, 0.004940, -0.452984,
]);
const NUM_JOINTS = 14;
const OBS_SIZE = 61;
const CMD_SIZE = 13;
const ACTION_SCALE = 1.0;
const TIMESTEP = 0.005;
const DECIMATION = 4;
const CTRL_DT = TIMESTEP * DECIMATION; // 50 Hz

// Velocity command limits, same as infer_policy.py's keyboard mapping.
const VEL_FWD = 0.25, VEL_BACK = -0.2, VEL_LAT = 0.2, VEL_ANG = 1.0;

const mount = document.getElementById("scene");
const loadingEl = document.getElementById("loading");
const hudEl = document.getElementById("hud");
const statsEl = document.getElementById("stats");
const setLoading = (msg) => { loadingEl.textContent = msg; };

// Surface boot failures on the page itself: a rejected top-level await
// otherwise leaves the loading screen up with no visible error.
window.addEventListener("unhandledrejection", (e) => {
  const msg = e.reason?.stack || e.reason?.message || String(e.reason);
  setLoading(`Boot failed: ${msg}`);
  console.error("[rl] unhandled rejection", e.reason);
});
window.addEventListener("error", (e) => {
  setLoading(`Boot failed: ${e.message}`);
});
const bootSteps = [];
const traced = (label, p) => {
  bootSteps.push(label);
  return p.then(
    (v) => {
      bootSteps.splice(bootSteps.indexOf(label), 1);
      if (bootSteps.length) setLoading(`Loading ${bootSteps.join(", ")}\u2026`);
      return v;
    },
    (err) => { console.error(`[rl] ${label} FAILED`, err); throw err; },
  );
};

// ── MJCF preparation ────────────────────────────────────────────────────
// robot_allcollisions.xml is what infer_policy.py's scene.xml includes: it
// carries body/shell collision geoms that robot_walk.xml lacks, which the
// sitstand policy needs (a sit rests the trunk on the ground).
// The visual meshes are irrelevant to the dynamics: every body carries an
// explicit <inertial>, and visual geoms have contype=0 conaffinity=0.
// Stripping them means the MuJoCo VFS only needs the ~10 meshes referenced
// by collision geoms.
async function buildPhysicsXml() {
  const src = await (await fetch(signed(`${MODEL_DIR}/robot_allcollisions.xml`))).text();
  const doc = new DOMParser().parseFromString(src, "text/xml");
  for (const g of [...doc.querySelectorAll('geom[class="visual"]')]) g.remove();
  const usedMeshes = new Set(
    [...doc.querySelectorAll("geom[mesh]")].map((g) => g.getAttribute("mesh")),
  );
  for (const m of [...doc.querySelectorAll("asset > mesh")]) {
    const name = m.getAttribute("name") ?? m.getAttribute("file").replace(/\.stl$/i, "");
    if (!usedMeshes.has(name)) m.remove();
  }
  const root = doc.documentElement;
  const el = (tag, attrs) => {
    const e = doc.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    return e;
  };
  root.appendChild(el("option", { timestep: String(TIMESTEP) }));
  doc.querySelector("worldbody").appendChild(
    el("geom", { name: "floor", type: "plane", size: "0 0 0.05", pos: "0 0 0" }),
  );
  // STAND keyframe from mjlab's scene_walk.xml (STAND2 pose).
  const qposFree = "0 0 0.12 1 0 0 0";
  const pose14 = Array.from(DEFAULT_POSE).join(" ");
  const kf = doc.createElement("keyframe");
  kf.appendChild(el("key", { name: "STAND", qpos: `${qposFree} ${pose14}`, ctrl: pose14 }));
  root.appendChild(kf);
  const meshFiles = [...doc.querySelectorAll("asset > mesh")].map((m) => m.getAttribute("file"));
  return { xml: new XMLSerializer().serializeToString(doc), meshFiles };
}

// ── Boot physics + policy in parallel with the render rig ──────────────
setLoading("Loading MuJoCo WASM, policies and meshes\u2026");

const [mujoco, { xml, meshFiles }, k] = await Promise.all([
  traced("mujoco wasm", loadMujoco()),
  traced("physics xml", buildPhysicsXml()),
  traced("kinematics", loadKinematics(`${MODEL_DIR}/kinematics.json`)),
]);

const vfs = new mujoco.MjVFS();
await Promise.all(
  meshFiles.map(async (f) => {
    // Same cache-busted URL as duck.js so the browser reuses the render
    // meshes instead of downloading the collision subset a second time.
    const buf = await (await fetch(signed(`${MODEL_DIR}/meshes/${f}?v=${MESH_VERSION}`), { cache: "force-cache" })).arrayBuffer();
    // meshdir="assets" in the MJCF, so the compiler looks up "assets/<f>".
    vfs.addBuffer(`assets/${f}`, new Uint8Array(buf));
  }),
);

const sessions = {};
let currentVariant = randomVariantName();
const rigPromise = (async () => {
  return buildRig(k, { materialForMesh: materialHookFor(VARIANTS[currentVariant]) });
})();
const sessionOpts = { executionProviders: ["wasm"] };
[sessions.walk, sessions.sitstand, sessions.roulade] = await Promise.all([
  ort.InferenceSession.create(signed(POLICIES.walk), sessionOpts),
  ort.InferenceSession.create(signed(POLICIES.sitstand), sessionOpts),
  ort.InferenceSession.create(signed(POLICIES.roulade), sessionOpts),
]);

setLoading("Compiling physics\u2026");
const model = mujoco.MjModel.from_xml_string(xml, vfs);
const data = new mujoco.MjData(model);

// Addresses resolved once. qpos/qvel/sensordata views are re-read at each
// use: the WASM heap can grow and detach earlier TypedArray views.
// NOTE: unlike the Python bindings, these accessor fields are plain numbers.
const qposAdr = JOINT_NAMES.map((n) => model.jnt(n).qposadr);
const dofAdr = JOINT_NAMES.map((n) => model.jnt(n).dofadr);
const gyroAdr = model.sensor("imu_ang_vel").adr;
const trunkId = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, "trunk_base");
const standKeyId = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY.value, "STAND");

let uiReady = false;
const lastAction = new Float32Array(NUM_JOINTS);
const obs = new Float32Array(OBS_SIZE);
const cmd = new Float32Array(CMD_SIZE); // [vx, vy, wz, head(4), body(6)]
const velCmd = new Float32Array(3); // twist command, driven by held keys
// Gamepad twist (EMA-smoothed like the robot runtime). Declared up here:
// the control loop reads it before the input section below has evaluated.
const padCmd = new Float32Array(3);
let padActive = false;
// Declared before the control loop starts: buildObs checks it to decide
// between the auto-run default and manual key control.
const held = new Set();

let mode = "walk"; // "walk" | "sitstand" | "roulade"
let sitFlag = 0;
// One-shot roulade tracking: trigger time + whether the trunk actually
// tipped over yet. Set by triggerRoulade, cleared when we hand back to walk.
let rouladeRun = null;

function resetSim() {
  mujoco.mj_resetDataKeyframe(model, data, standKeyId);
  mujoco.mj_forward(model, data);
  lastAction.fill(0);
  sitFlag = 0;
  // Buttons reflect sitFlag; keep them honest after auto-resets.
  if (uiReady) syncButtons();
}
resetSim();

// ── Observation ─────────────────────────────────────────────────────────
const _q = new THREE.Quaternion();
const _g = new THREE.Vector3();

function buildObs() {
  const qpos = data.qpos, qvel = data.qvel, sens = data.sensordata;
  let i = 0;
  // base_ang_vel: gyro sensor at the IMU site
  for (let a = 0; a < 3; a++) obs[i++] = sens[gyroAdr + a];
  // projected gravity: world -z rotated into the trunk frame
  const xq = data.body(trunkId).xquat; // [w, x, y, z]
  _q.set(xq[1], xq[2], xq[3], xq[0]).conjugate();
  _g.set(0, 0, -1).applyQuaternion(_q);
  obs[i++] = _g.x; obs[i++] = _g.y; obs[i++] = _g.z;
  for (let j = 0; j < NUM_JOINTS; j++) obs[i++] = qpos[qposAdr[j]] - DEFAULT_POSE[j];
  for (let j = 0; j < NUM_JOINTS; j++) obs[i++] = qvel[dofAdr[j]];
  for (let j = 0; j < NUM_JOINTS; j++) obs[i++] = lastAction[j];
  // command: walking/roulade use the twist; sitstand uses cmd[0] as the
  // posture flag. "Run" means run: with no keys held the walking policy
  // gets a forward velocity by default, keys override it.
  cmd.fill(0, 0, 3);
  if (mode === "walk") {
    if (padActive) {
      cmd[0] = padCmd[0]; cmd[1] = padCmd[1]; cmd[2] = padCmd[2];
    } else {
      cmd[0] = held.size ? velCmd[0] : VEL_FWD;
      cmd[1] = velCmd[1]; cmd[2] = velCmd[2];
    }
  } else if (mode === "roulade") {
    cmd[0] = velCmd[0]; cmd[1] = velCmd[1]; cmd[2] = velCmd[2];
  } else {
    cmd[0] = sitFlag;
  }
  for (let c = 0; c < CMD_SIZE; c++) obs[i++] = cmd[c];
  return obs;
}

// ── Control loop (50 Hz, async because ONNX inference is async) ────────
let ctrlHz = 0;
let fallenSince = null;

async function controlStep() {
  const feeds = { obs: new ort.Tensor("float32", buildObs(), [1, OBS_SIZE]) };
  const out = await sessions[mode].run(feeds);
  const act = out.actions.data;
  lastAction.set(act);
  const ctrl = data.ctrl;
  for (let j = 0; j < NUM_JOINTS; j++) ctrl[j] = DEFAULT_POSE[j] + act[j] * ACTION_SCALE;
  for (let s = 0; s < DECIMATION; s++) mujoco.mj_step(model, data);

  // Fall detection: walk/sitstand have no get-up skill, so auto-reset when
  // down for good. Height alone would false-positive on a deep sit, so
  // "fallen" = trunk tilted past ~60 deg (projected gravity z above -0.5)
  // or sunk below the floor, sustained for over a second. The roulade rolls
  // the trunk on purpose and recovers on its own, so it gets a much longer
  // grace window before we call it stuck.
  const z = data.qpos[2];
  const now = performance.now();
  const tipped = obs[5] > -0.5; // projected gravity z, from the last obs
  const graceMs = mode === "roulade" ? 5000 : 1000;
  if (tipped || z < 0.02) {
    fallenSince ??= now;
    if (now - fallenSince > graceMs) { resetSim(); fallenSince = null; }
  } else {
    fallenSince = null;
  }

  // One-shot roulade, step-counted like the robot runtime (a single roll is
  // ~1 s = 50 control steps there): hand back to walking once the trunk has
  // tipped over and is upright again, or after a hard 2 s window if the roll
  // never initiated. Counting steps instead of wall time keeps the logic
  // correct when the sim is fast-forwarded or the tab is throttled.
  if (mode === "roulade" && rouladeRun) {
    rouladeRun.steps++;
    if (obs[5] > -0.3) rouladeRun.tipped = true;
    const upright = obs[5] < -0.85;
    const done = rouladeRun.tipped && upright && rouladeRun.steps >= 40;
    const expired = rouladeRun.steps >= 150; // 3 s, roll should long be over
    if (done || expired) {
      rouladeRun = null;
      mode = "walk";
      lastAction.fill(0);
      // Timed out mid-roll: don't hand a tipped duck to the walking policy
      // (it has no get-up skill), restart from the keyframe instead.
      if (!upright) resetSim();
      if (uiReady) syncButtons();
    }
  }
}

let running = true;
(async function controlLoop() {
  let next = performance.now();
  let count = 0, hzT0 = next;
  while (running) {
    await controlStep();
    count++;
    const now = performance.now();
    if (now - hzT0 > 500) {
      ctrlHz = (count * 1000) / (now - hzT0);
      count = 0; hzT0 = now;
    }
    next += CTRL_DT * 1000;
    const wait = next - performance.now();
    if (wait > 0) await new Promise((r) => setTimeout(r, wait));
    else next = performance.now(); // fell behind: don't spiral
  }
})();

// ── Rendering (three.js rig driven by qpos) ─────────────────────────────
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x08080c);

const camera = new THREE.PerspectiveCamera(40, 1, 0.02, 30);
camera.position.set(0.55, 0.35, 0.7);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
mount.appendChild(renderer.domElement);

const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment()).texture;
scene.environmentIntensity = 0.45;

scene.add(new THREE.AmbientLight(0xffffff, 0.6));
const keyLight = new THREE.DirectionalLight(0xffffff, 1.6);
keyLight.position.set(2, 4, 2);
scene.add(keyLight);
const fill = new THREE.DirectionalLight(0xffffff, 0.4);
fill.position.set(-2, 2, 1.5);
scene.add(fill);
const rim = new THREE.DirectionalLight(0xffb366, 0.7);
rim.position.set(0, 3, -2);
scene.add(rim);

// Infinite shader grid, ported from drei's <Grid> (we're in vanilla three,
// not R3F): anti-aliased world-space lines at cell/section frequencies with
// a radial fade around the duck. Lines derive from world coordinates, so
// re-centering the mesh under the camera target every frame makes the grid
// effectively infinite without any visible swimming.
function makeInfiniteGrid() {
  const material = new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    uniforms: {
      uCell: { value: 0.1 },
      uSection: { value: 0.5 },
      uCellColor: { value: new THREE.Color(0x7d7360) },
      uSectionColor: { value: new THREE.Color(0xffb366) },
      uFadeDist: { value: 3.0 },
      uFocus: { value: new THREE.Vector3() },
    },
    vertexShader: /* glsl */ `
      varying vec3 vWorld;
      void main() {
        vec4 w = modelMatrix * vec4(position, 1.0);
        vWorld = w.xyz;
        gl_Position = projectionMatrix * viewMatrix * w;
      }
    `,
    fragmentShader: /* glsl */ `
      varying vec3 vWorld;
      uniform float uCell, uSection, uFadeDist;
      uniform vec3 uCellColor, uSectionColor, uFocus;
      float gridLine(vec2 p, float size) {
        vec2 r = p / size;
        vec2 g = abs(fract(r - 0.5) - 0.5) / fwidth(r);
        return 1.0 - min(min(g.x, g.y), 1.0);
      }
      void main() {
        float cell = gridLine(vWorld.xz, uCell);
        float section = gridLine(vWorld.xz, uSection);
        float d = distance(vWorld.xz, uFocus.xz);
        float fade = pow(clamp(1.0 - d / uFadeDist, 0.0, 1.0), 1.6);
        vec3 col = mix(uCellColor, uSectionColor, section);
        float alpha = max(section * 0.45, cell * 0.3) * fade;
        if (alpha < 0.004) discard;
        gl_FragColor = vec4(col, alpha);
      }
    `,
  });
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(30, 30), material);
  mesh.rotation.x = -Math.PI / 2;
  return mesh;
}
const grid = makeInfiniteGrid();
scene.add(grid);

const rig = await rigPromise;
scene.add(rig.placer);
const trunkGroup = rig.bodies.get("trunk_base");

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 0.25;
controls.maxDistance = 3;
controls.maxPolarAngle = Math.PI / 2 - 0.03;

function resize() {
  const w = mount.clientWidth, h = mount.clientHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
}
new ResizeObserver(resize).observe(mount);
resize();

// The rig's root already applies the MJCF Z-up -> three Y-up fix, so the
// trunk group can take the freejoint pose in raw MJCF coordinates.
const _target = new THREE.Vector3();
const _follow = new THREE.Vector3();
function syncRig() {
  const qpos = data.qpos;
  trunkGroup.position.set(qpos[0], qpos[1], qpos[2]);
  trunkGroup.quaternion.set(qpos[4], qpos[5], qpos[6], qpos[3]);
  for (let j = 0; j < NUM_JOINTS; j++) setJoint(rig, JOINT_NAMES[j], qpos[qposAdr[j]]);
  // Follow cam: ease the orbit target toward the trunk and translate the
  // camera by the same delta, so the camera-to-duck distance and viewing
  // angle stay constant while the duck walks. Mouse orbit/zoom still work:
  // they only change the (preserved) camera-target offset.
  _target.set(qpos[0], qpos[2], -qpos[1]);
  _follow.copy(_target).sub(controls.target).multiplyScalar(0.06);
  controls.target.add(_follow);
  camera.position.add(_follow);
  // Keep the grid plane (and its fade center) under the action.
  grid.position.set(controls.target.x, 0, controls.target.z);
  grid.material.uniforms.uFocus.value.copy(controls.target);
}

// Quack: a quick jaw flap on every mode/colour change. The jaw isn't a
// MuJoCo joint (duck.js re-creates the hinge in JS), so this is purely
// cosmetic and can't upset the policy. A held gamepad trigger drives the
// jaw analogically on top (same as the robot's mouth trigger).
const QUACK_MS = 480;
let quackAt = -Infinity;
let padJaw = 0;
const quack = () => { quackAt = performance.now(); };
function syncJaw() {
  const t = (performance.now() - quackAt) / QUACK_MS;
  const flap = t >= 0 && t < 1 ? Math.sin(Math.PI * t) : 0;
  setJawOpen(rig, Math.max(flap, padJaw));
}

function renderStats() {
  const [vx, vy, wz] = padActive ? padCmd : velCmd;
  const posture = mode === "walk"
    ? `running policy`
    : mode === "roulade"
      ? `roulade policy`
      : `sitstand policy \u00b7 ${sitFlag ? "sit" : "stand"}`;
  statsEl.innerHTML =
    `<b>${posture}</b><br>` +
    `cmd vx ${vx.toFixed(2)} \u00b7 vy ${vy.toFixed(2)} \u00b7 wz ${wz.toFixed(2)}<br>` +
    `ctrl ${ctrlHz.toFixed(0)} Hz \u00b7 sim t ${data.time.toFixed(1)} s`;
}

// Real implementation assigned in the gamepad section below; the render
// loop starts before that section has evaluated, hence the indirection.
let pollPad = () => {};

function loop() {
  requestAnimationFrame(loop);
  pollPad();
  syncRig();
  syncJaw();
  controls.update();
  renderStats();
  renderer.render(scene, camera);
}
loadingEl.style.display = "none";
hudEl.hidden = false;
loop();

// ── Input: hold-to-command keys + HUD buttons ───────────────────────────
function refreshVelCmd() {
  velCmd[0] = held.has("fwd") ? VEL_FWD : held.has("back") ? VEL_BACK : 0;
  velCmd[1] = held.has("left") ? VEL_LAT : held.has("right") ? -VEL_LAT : 0;
  velCmd[2] = held.has("turnl") ? VEL_ANG : held.has("turnr") ? -VEL_ANG : 0;
}

const KEYMAP = {
  ArrowUp: "fwd", KeyW: "fwd", KeyZ: "fwd",
  ArrowDown: "back", KeyS: "back",
  ArrowLeft: "turnl", ArrowRight: "turnr",
  KeyA: "left", KeyQ: "left", KeyE: "right", KeyD: "right",
};

window.addEventListener("keydown", (e) => {
  if (e.repeat) return;
  if (e.code === "KeyR") { resetSim(); return; }
  if (e.code === "Space") { e.preventDefault(); triggerRoulade(); return; }
  const act = KEYMAP[e.code];
  if (!act) return;
  e.preventDefault();
  held.add(act);
  refreshVelCmd();
});
window.addEventListener("keyup", (e) => {
  const act = KEYMAP[e.code];
  if (!act) return;
  held.delete(act);
  refreshVelCmd();
});
window.addEventListener("blur", () => { held.clear(); refreshVelCmd(); });

// ── Gamepad: same mapping as the robot runtime (microduck_runtime) ──────
// Sticks: L vertical = vx (asymmetric fwd/back), L horizontal = strafe,
// R horizontal = turn, all EMA-smoothed like the runtime's cmd_alpha.
// X = roulade, DpadDown = sit/stand toggle, RT = mouth (analog) + quack.
const PAD_DEADZONE = 0.15;
const PAD_ALPHA = 0.12;
const padPrev = { x: false, dpadDown: false, rt: 0 };
const dz = (v) => (Math.abs(v) < PAD_DEADZONE ? 0 : v);

pollPad = function pollGamepad() {
  const gp = [...(navigator.getGamepads?.() ?? [])].find((p) => p && p.connected);
  document.body.classList.toggle("pad-connected", !!gp);
  if (!gp) {
    if (padActive) { padActive = false; padCmd.fill(0); padJaw = 0; }
    return;
  }
  const lx = dz(gp.axes[0] ?? 0), ly = dz(gp.axes[1] ?? 0), rx = dz(gp.axes[2] ?? 0);
  const up = -ly; // browser sticks report up as -1
  const target = [
    up >= 0 ? up * VEL_FWD : up * -VEL_BACK,
    -lx * VEL_LAT,
    -rx * VEL_ANG,
  ];
  for (let i = 0; i < 3; i++) padCmd[i] += PAD_ALPHA * (target[i] - padCmd[i]);
  // Sticks grab command authority on first input, release when back at rest
  // (then keyboard / auto-run take over again).
  const stickInput = lx !== 0 || ly !== 0 || rx !== 0;
  if (stickInput) padActive = true;
  else if (padActive && Math.abs(padCmd[0]) + Math.abs(padCmd[1]) + Math.abs(padCmd[2]) < 0.01) {
    padActive = false;
    padCmd.fill(0);
  }

  // Standard mapping indices: X=2, DpadDown=13, RT=7 (analog value).
  const x = !!gp.buttons[2]?.pressed;
  if (x && !padPrev.x) triggerRoulade();
  padPrev.x = x;

  const dpadDown = !!gp.buttons[13]?.pressed;
  if (dpadDown && !padPrev.dpadDown) {
    const sitting = mode === "sitstand" && sitFlag === 1;
    setMode(sitting ? "walk" : "sit");
  }
  padPrev.dpadDown = dpadDown;

  const rt = gp.buttons[7]?.value ?? 0;
  padJaw = rt;
  if (padPrev.rt < 0.3 && rt >= 0.3) quack();
  padPrev.rt = rt;
};

const btnWalk = document.getElementById("btn-walk");
const btnSit = document.getElementById("btn-sit");
const btnRoulade = document.getElementById("btn-roulade");

function setMode(next) {
  quack();
  rouladeRun = null;
  if (next !== "sit") {
    // Leaving a sit: let the sitstand policy stand the duck back up first.
    if (mode === "sitstand" && sitFlag === 1) {
      sitFlag = 0;
      setTimeout(() => { mode = next; lastAction.fill(0); syncButtons(); }, 2000);
      syncButtons();
      return;
    }
    mode = next;
    lastAction.fill(0);
  } else {
    // Hand over gently: hold the stand under the sitstand policy for a
    // moment before commanding the sit, or the abrupt session switch
    // (walking's action history + instant flag) knocks the duck over.
    mode = "sitstand";
    sitFlag = 0;
    lastAction.fill(0);
    setTimeout(() => {
      if (mode === "sitstand") { sitFlag = 1; syncButtons(); }
    }, 800);
  }
  syncButtons();
}

// One roulade, then straight back to running (Space key or the button).
// lastAction is deliberately NOT zeroed here: the robot runtime keeps one
// continuous action history across policy switches, and the roll initiates
// more reliably mid-gait with the true last actions in the obs.
function triggerRoulade() {
  if (mode === "roulade") return;
  quack();
  mode = "roulade";
  sitFlag = 0;
  rouladeRun = { steps: 0, tipped: false };
  syncButtons();
}
function syncButtons() {
  const sitting = mode === "sitstand" && sitFlag === 1;
  btnWalk.classList.toggle("on", mode === "walk" || (mode === "sitstand" && !sitting));
  btnSit.classList.toggle("on", sitting);
  btnRoulade.classList.toggle("on", mode === "roulade");
}
btnWalk.addEventListener("click", () => setMode("walk"));
btnSit.addEventListener("click", () => setMode("sit"));
btnRoulade.addEventListener("click", triggerRoulade);

// ── Colour swatches: re-skin the rig live, with a quack ─────────────────
// One representative colour per variant so the dots read at a glance.
const SWATCH_SLOT = { classic: "feet", charcoal: "headDome", purple: "feet", blue: "facePlate" };
const swatchesEl = document.getElementById("swatches");
const swatchBtns = new Map();
for (const name of VARIANT_NAMES) {
  const v = VARIANTS[name];
  const b = document.createElement("button");
  b.style.background = specToHex(v[SWATCH_SLOT[name] ?? "bodyShell"]);
  b.setAttribute("aria-label", `${name} colours`);
  b.addEventListener("click", () => {
    if (name === currentVariant) return;
    currentVariant = name;
    applyVariant(rig, name);
    quack();
    syncSwatches();
  });
  swatchesEl.appendChild(b);
  swatchBtns.set(name, b);
}
function syncSwatches() {
  for (const [name, b] of swatchBtns) b.classList.toggle("on", name === currentVariant);
}
syncSwatches();
syncButtons();
uiReady = true;

// Deterministic hooks for automated verification (rAF pauses in
// background tabs, and the control loop is async).
window.rl = {
  model, data, mujoco,
  get mode() { return mode; },
  get sitFlag() { return sitFlag; },
  buildObs, cmd,
  velCmd, lastAction, resetSim,
  step: async (n = 1) => { for (let i = 0; i < n; i++) await controlStep(); },
  render: () => { syncRig(); renderer.render(scene, camera); },
};
