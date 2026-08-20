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
import { buildRig, loadKinematics, setJoint, setJawOpen, MODEL_DIR, MESH_VERSION } from "./duck.js";
import { VARIANTS, VARIANT_NAMES, materialHookFor, randomVariantName, applyVariant, specToHex } from "./variants.js";
import loadMujoco from "https://cdn.jsdelivr.net/npm/@mujoco/mujoco@3.11.0/mujoco.js";
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/ort.min.mjs";

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
const verbEl = document.getElementById("verb");
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
  const src = await (await fetch(`${MODEL_DIR}/robot_allcollisions.xml`)).text();
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
    const buf = await (await fetch(`${MODEL_DIR}/meshes/${f}?v=${MESH_VERSION}`, { cache: "force-cache" })).arrayBuffer();
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
  ort.InferenceSession.create(POLICIES.walk, sessionOpts),
  ort.InferenceSession.create(POLICIES.sitstand, sessionOpts),
  ort.InferenceSession.create(POLICIES.roulade, sessionOpts),
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
// Declared before the control loop starts: buildObs checks it to decide
// between the auto-run default and manual key control.
const held = new Set();

let mode = "walk"; // "walk" | "sitstand" | "roulade"
let sitFlag = 0;

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
    cmd[0] = held.size ? velCmd[0] : VEL_FWD;
    cmd[1] = velCmd[1]; cmd[2] = velCmd[2];
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

function makeFloorTexture() {
  const N = 256;
  const c = document.createElement("canvas");
  c.width = c.height = N;
  const g = c.getContext("2d");
  g.fillStyle = "#131009";
  g.fillRect(0, 0, N, N);
  g.strokeStyle = "rgba(255, 179, 102, 0.05)";
  g.lineWidth = 2;
  g.strokeRect(1, 1, N - 2, N - 2);
  g.fillStyle = "rgba(255, 198, 30, 0.10)";
  for (const x of [0, N]) {
    for (const y of [0, N]) {
      g.beginPath();
      g.arc(x, y, 5, 0, Math.PI * 2);
      g.fill();
    }
  }
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(48, 48);
  return tex;
}
const floor = new THREE.Mesh(
  new THREE.PlaneGeometry(12, 12),
  new THREE.MeshStandardMaterial({ map: makeFloorTexture(), roughness: 0.9, metalness: 0.0 }),
);
floor.rotation.x = -Math.PI / 2;
scene.add(floor);

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
function syncRig() {
  const qpos = data.qpos;
  trunkGroup.position.set(qpos[0], qpos[1], qpos[2]);
  trunkGroup.quaternion.set(qpos[4], qpos[5], qpos[6], qpos[3]);
  for (let j = 0; j < NUM_JOINTS; j++) setJoint(rig, JOINT_NAMES[j], qpos[qposAdr[j]]);
  // Camera target eases toward the trunk so the duck stays framed while
  // it walks away.
  _target.set(qpos[0], qpos[2], -qpos[1]);
  controls.target.lerp(_target, 0.06);
}

// Quack: a quick jaw flap on every mode/colour change. The jaw isn't a
// MuJoCo joint (duck.js re-creates the hinge in JS), so this is purely
// cosmetic and can't upset the policy.
const QUACK_MS = 480;
let quackAt = -Infinity;
const quack = () => { quackAt = performance.now(); };
function syncJaw() {
  const t = (performance.now() - quackAt) / QUACK_MS;
  setJawOpen(rig, t >= 0 && t < 1 ? Math.sin(Math.PI * t) : 0);
}

function renderStats() {
  const [vx, vy, wz] = velCmd;
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

function loop() {
  requestAnimationFrame(loop);
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

const btnWalk = document.getElementById("btn-walk");
const btnSit = document.getElementById("btn-sit");
const btnRoulade = document.getElementById("btn-roulade");
document.getElementById("btn-reset").addEventListener("click", resetSim);

// The headline verb tracks the active policy.
const VERBS = { walk: "run", sitstand: "sit", roulade: "roll over" };

function setMode(next) {
  quack();
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
function syncButtons() {
  const sitting = mode === "sitstand" && sitFlag === 1;
  btnWalk.classList.toggle("on", mode === "walk" || (mode === "sitstand" && !sitting));
  btnSit.classList.toggle("on", sitting);
  btnRoulade.classList.toggle("on", mode === "roulade");
  verbEl.textContent = VERBS[mode];
}
btnWalk.addEventListener("click", () => setMode("walk"));
btnSit.addEventListener("click", () => setMode("sit"));
btnRoulade.addEventListener("click", () => setMode("roulade"));

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
