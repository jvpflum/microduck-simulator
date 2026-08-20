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
// They also reuse rl.js's own ?v= cache-buster: python http.server (and
// some CDN edges) send no Cache-Control, so without it browsers can keep
// serving stale module bytes after a deploy.
const SELF_V = new URL(import.meta.url).searchParams.get("v") ?? "0";
const { buildRig, cloneRig, loadKinematics, setJoint, setJawOpen, MODEL_DIR, MESH_VERSION } =
  await import(signed(`./duck.js?v=${SELF_V}`));
const { VARIANTS, VARIANT_NAMES, materialHookFor, randomVariantName, applyVariant, specToHex } =
  await import(signed(`./variants.js?v=${SELF_V}`));

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/";
ort.env.wasm.numThreads = 1; // static hosting sends no COOP/COEP headers

const POLICY_DIR = "./policies";
const POLICIES = {
  walk: `${POLICY_DIR}/BEST_alpha_walking.onnx`,
  sitstand: `${POLICY_DIR}/BEST_alpha_sitstand.onnx`,
  roll: `${POLICY_DIR}/roulade.onnx`,
  // Blind one-shot kicks (the operator aims the robot, no ball in obs):
  // the runtime swaps these in for a 0.5 s window, commands zeroed.
  kickL: `${POLICY_DIR}/ball_kick_left.onnx`,
  kickR: `${POLICY_DIR}/ball_kick_right.onnx`,
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

// Kickable ball: radius, parking spot (far away = hidden by default) and
// the distance past which the respawn watchdog brings it back (the grid
// fades out at 3 m, so a ball beyond that is invisible anyway).
const BALL_RADIUS = 0.05;
const BALL_PARK_POS = "50 0 0.05";
const BALL_MAX_DIST = 3.0;

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
  // Kickable ball: a light free sphere (beach-ball feel). MuJoCo has no
  // restitution parameter - the bounce comes from solref damping < 1, and
  // the rolling-friction term makes it come to rest. Default contype /
  // conaffinity (1) collide with the floor and the duck's collision-class
  // geoms; the self_collision_only class (contype=2) correctly ignores it.
  // Appended AFTER the robot body so the trunk freejoint stays first in
  // qpos: qpos[0..6] indexing is hardcoded in syncRig, the fall watchdog
  // and the ghosts' getLocalState.
  const ballBody = el("body", { name: "ball", pos: BALL_PARK_POS });
  ballBody.appendChild(el("freejoint", { name: "ball_freejoint" }));
  // condim 6 enables the torsional + rolling friction components of the
  // friction vector; with the default condim 3 they are ignored and a
  // rolling ball never decelerates.
  ballBody.appendChild(el("geom", {
    name: "ball_geom", type: "sphere", size: String(BALL_RADIUS),
    mass: "0.03", friction: "0.4 0.01 0.003", solref: "0.03 0.4", condim: "6",
  }));
  doc.querySelector("worldbody").appendChild(ballBody);
  // STAND keyframe from mjlab's scene_walk.xml (STAND2 pose). The ball's
  // 7 free-joint values MUST be appended or nq (21 + 7 = 28) won't match
  // and the model won't compile; parked 50 m away = effectively absent.
  const qposFree = "0 0 0.12 1 0 0 0";
  const pose14 = Array.from(DEFAULT_POSE).join(" ");
  const kf = doc.createElement("keyframe");
  kf.appendChild(el("key", {
    name: "STAND",
    qpos: `${qposFree} ${pose14} ${BALL_PARK_POS} 1 0 0 0`,
    ctrl: pose14,
  }));
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
[sessions.walk, sessions.sitstand, sessions.roll, sessions.kickL, sessions.kickR] =
  await Promise.all([
    ort.InferenceSession.create(signed(POLICIES.walk), sessionOpts),
    ort.InferenceSession.create(signed(POLICIES.sitstand), sessionOpts),
    ort.InferenceSession.create(signed(POLICIES.roll), sessionOpts),
    ort.InferenceSession.create(signed(POLICIES.kickL), sessionOpts),
    ort.InferenceSession.create(signed(POLICIES.kickR), sessionOpts),
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
const ballQposAdr = model.jnt("ball_freejoint").qposadr;
const ballDofAdr = model.jnt("ball_freejoint").dofadr;

let uiReady = false;
const lastAction = new Float32Array(NUM_JOINTS);
const obs = new Float32Array(OBS_SIZE);
const cmd = new Float32Array(CMD_SIZE); // [vx, vy, wz, head(4), body(6)]
const velCmd = new Float32Array(3); // twist command, driven by held keys
// Gamepad twist (EMA-smoothed like the robot runtime). Declared up here:
// the control loop reads it before the input section below has evaluated.
const padCmd = new Float32Array(3);
let padActive = false;
const padPrev = { x: false, y: false, rb: false, lb: false, dpadDown: false, dpadUp: false, rtArmed: true };
// Declared before the control loop starts: buildObs reads it via
// effectiveCmd on the very first control step.
const held = new Set();

let mode = "walk"; // "walk" | "sitstand" | "roll" | "kickL" | "kickR"
let sitFlag = 0;
const isKick = () => mode === "kickL" || mode === "kickR";
// Local-only kickable ball: false while parked at the keyframe spot
// (mesh hidden), true once popped in front of the duck.
let ballActive = false;

// The twist the policy actually receives: live gamepad sticks win over the
// held keys. No input means zero command - the duck stands in place.
// Shared by buildObs and the HUD mini-sticks so they can never disagree.
// Mid-roll every movement input is ignored (zero twist) until the roll
// hands back to walk on its own - steering would only knock the roll over.
const ZERO_CMD = new Float32Array(3);
function effectiveCmd() {
  if (mode === "roll" || isKick()) return ZERO_CMD;
  return padActive ? padCmd : velCmd;
}
// One-shot roll tracking: trigger time + whether the trunk actually
// tipped over yet. Set by triggerRoll, cleared when we hand back to walk.
let rollRun = null;
let rollSource = "kb"; // which hint lights up: keyboard Space or pad X
// One-shot kick tracking: the runtime swaps the kick policy in for a fixed
// 0.5 s window (25 control steps at 50 Hz) with zeroed commands, then hands
// straight back to walking. lastAction stays continuous across both swaps.
let kickRun = null;
let kickSource = "kb";
const KICK_STEPS = 25;

// Pending mode-transition timers (sit hand-over, stand-up hand-back).
// Every transition entry point clears them: a stale timer firing after
// the state has moved on is exactly how a sitting or rolling duck ends
// up handed to the walking policy mid-motion.
let sitTimer = null;
let standTimer = null;
function clearModeTimers() {
  clearTimeout(sitTimer); sitTimer = null;
  clearTimeout(standTimer); standTimer = null;
}

function resetSim() {
  // A reset is a full transition to the walking stand: cancel any roll in
  // flight and any pending sit/stand hand-over, or they'd replay on the
  // freshly reset duck.
  clearModeTimers();
  rollRun = null;
  kickRun = null;
  mode = "walk";
  mujoco.mj_resetDataKeyframe(model, data, standKeyId);
  mujoco.mj_forward(model, data);
  lastAction.fill(0);
  sitFlag = 0;
  // The keyframe re-parks the ball 50 m away; reflect that in the flag
  // so the render loop hides the mesh again.
  ballActive = false;
  // Buttons reflect sitFlag; keep them honest after auto-resets.
  if (uiReady) syncButtons();
}
resetSim();

// Pop / respawn the ball ~0.35 m in front of the duck, with a small
// random heading + distance jitter so repeated pops land somewhere
// nearby instead of always on the same spot. qvel is zeroed so a respawn
// doesn't carry the old momentum.
function spawnBall() {
  const qpos = data.qpos, qvel = data.qvel;
  // Trunk yaw from the free-joint quaternion (qpos[3..6] = w x y z);
  // the duck walks toward its local +X.
  const yaw = Math.atan2(
    2 * (qpos[3] * qpos[6] + qpos[4] * qpos[5]),
    1 - 2 * (qpos[5] * qpos[5] + qpos[6] * qpos[6]),
  );
  const heading = yaw + (Math.random() - 0.5) * 0.7;
  const dist = 0.35 + (Math.random() - 0.5) * 0.1;
  qpos[ballQposAdr] = qpos[0] + Math.cos(heading) * dist;
  qpos[ballQposAdr + 1] = qpos[1] + Math.sin(heading) * dist;
  qpos[ballQposAdr + 2] = BALL_RADIUS + 0.02;
  qpos[ballQposAdr + 3] = 1;
  qpos[ballQposAdr + 4] = 0;
  qpos[ballQposAdr + 5] = 0;
  qpos[ballQposAdr + 6] = 0;
  for (let i = 0; i < 6; i++) qvel[ballDofAdr + i] = 0;
  mujoco.mj_forward(model, data);
  ballActive = true;
}

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
  // command: walking/roll use the twist; sitstand uses cmd[0] as the
  // posture flag.
  cmd.fill(0, 0, 3);
  if (mode === "sitstand") {
    cmd[0] = sitFlag;
  } else {
    // walk uses the live twist; roll and kick see all-zero commands
    // (via effectiveCmd), matching how they were trained.
    const c = effectiveCmd();
    cmd[0] = c[0]; cmd[1] = c[1]; cmd[2] = c[2];
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
  // or sunk below the floor, sustained for over a second. The roll tumbles
  // the trunk on purpose and recovers on its own, so it gets a much longer
  // grace window before we call it stuck.
  const z = data.qpos[2];
  const now = performance.now();
  const tipped = obs[5] > -0.5; // projected gravity z, from the last obs
  const graceMs = mode === "roll" ? 5000 : 1000;
  if (tipped || z < 0.02) {
    fallenSince ??= now;
    if (now - fallenSince > graceMs) { resetSim(); fallenSince = null; }
  } else {
    fallenSince = null;
  }

  // Ball respawn watchdog: a kicked ball that wandered past the grid's
  // fade distance is invisible and unreachable - bring it back near
  // the duck. Re-read qpos: resetSim above may have re-parked the ball.
  if (ballActive) {
    const q = data.qpos;
    const dx = q[ballQposAdr] - q[0];
    const dy = q[ballQposAdr + 1] - q[1];
    if (dx * dx + dy * dy > BALL_MAX_DIST * BALL_MAX_DIST) spawnBall();
  }

  // One-shot roll, step-counted like the robot runtime (a single roll is
  // ~1 s = 50 control steps there): hand back to walking once the trunk has
  // tipped over and is upright again, or after a hard 2 s window if the roll
  // never initiated. Counting steps instead of wall time keeps the logic
  // correct when the sim is fast-forwarded or the tab is throttled.
  // One-shot kick: fixed 0.5 s window like the robot runtime, then straight
  // back to walking. lastAction is NOT zeroed on either swap - the runtime
  // keeps one continuous action history across policy switches.
  if (isKick() && kickRun) {
    kickRun.steps++;
    if (kickRun.steps >= KICK_STEPS) {
      kickRun = null;
      mode = "walk";
      if (uiReady) syncButtons();
    }
  }

  if (mode === "roll" && rollRun) {
    rollRun.steps++;
    if (obs[5] > -0.3) rollRun.tipped = true;
    const upright = obs[5] < -0.85;
    const done = rollRun.tipped && upright && rollRun.steps >= 40;
    const expired = rollRun.steps >= 150; // 3 s, roll should long be over
    if (done || expired) {
      rollRun = null;
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

// Soccer-ball look on an equirectangular CanvasTexture: off-white base
// (pure white would blow out under ACES tone mapping) with black
// pentagons - a cap at each pole plus two staggered rings - so the
// rolling actually reads visually.
function makeSoccerBallTexture() {
  const c = document.createElement("canvas");
  c.width = 1024;
  c.height = 512;
  const ctx = c.getContext("2d");
  ctx.fillStyle = "#e9e7e0";
  ctx.fillRect(0, 0, c.width, c.height);
  ctx.fillStyle = "#17171d";
  // Pole "pentagons": a full-width strip at each canvas edge maps to a
  // clean round patch at the pole, sidestepping the equirect pinch that
  // would smear an actual drawn polygon there.
  const capH = c.height * 0.075;
  ctx.fillRect(0, 0, c.width, capH);
  ctx.fillRect(0, c.height - capH, c.width, capH);
  // Pentagon at (lon, lat), angular radius r (all degrees). Horizontal
  // extent is stretched by 1/cos(lat) to counter the equirect longitude
  // compression away from the equator; drawn three times so panels
  // crossing the +-180 deg seam wrap around cleanly.
  const pent = (lonDeg, latDeg, rDeg, rot) => {
    const x = ((lonDeg + 180) / 360) * c.width;
    const y = ((90 - latDeg) / 180) * c.height;
    const ry = (rDeg / 180) * c.height;
    const rx = ry / Math.cos((latDeg * Math.PI) / 180);
    for (const dx of [-c.width, 0, c.width]) {
      ctx.beginPath();
      for (let i = 0; i < 5; i++) {
        const a = rot + (i * 2 * Math.PI) / 5;
        const px = x + dx + Math.cos(a) * rx;
        const py = y + Math.sin(a) * ry;
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.closePath();
      ctx.fill();
    }
  };
  // Two staggered rings of five (12 pentagons total with the poles),
  // roughly the truncated-icosahedron layout. Point-up above the
  // equator, point-down below, like the real panel orientation.
  for (let i = 0; i < 5; i++) {
    pent(-180 + i * 72, 27, 15, -Math.PI / 2);
    pent(-144 + i * 72, -27, 15, Math.PI / 2);
  }
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  // Texel footprints get extremely anamorphic near the UV poles; without
  // anisotropy the cap edge visibly scallops at close range.
  tex.anisotropy = renderer.capabilities.getMaxAnisotropy();
  return tex;
}
// Same Z-up -> Y-up trick as the duck rig: the group takes the axis fix,
// the mesh inside takes the raw MJCF free-joint pose.
const ballGroup = new THREE.Group();
ballGroup.rotation.x = -Math.PI / 2;
const ballMesh = new THREE.Mesh(
  // 48x32 segments: the coarser default makes the UV interpolation near
  // the poles visibly scallop the round cap edge of the texture.
  new THREE.SphereGeometry(BALL_RADIUS, 48, 32),
  new THREE.MeshStandardMaterial({ map: makeSoccerBallTexture(), roughness: 0.35, metalness: 0 }),
);
ballMesh.visible = false;
ballGroup.add(ballMesh);
scene.add(ballGroup);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 0.25;
controls.maxDistance = 3;
controls.maxPolarAngle = Math.PI / 2 - 0.03;

// Chase cam (default ON): each frame the camera eases toward a point
// behind the duck's heading at the current orbit distance, while the
// orbit target keeps easing to the trunk in syncRig. Implemented by
// overwriting camera.position AFTER controls.update() so we never fight
// OrbitControls' own spherical bookkeeping; the wheel still zooms in
// chase mode because the behind-point distance is re-read from the live
// camera-target distance every frame.
// Detach: any pointer grab on the canvas (drag start) drops back to the
// free orbit-follow. We listen on pointerdown rather than the controls'
// "start" event because in this three.js version the wheel dispatches
// "start" too, and scroll-to-zoom must not detach the chase.
let chaseCam = true;
const CHASE_PITCH = 0.42; // rad above horizontal, keeps the floor in view
const CHASE_EASE = 0.05; // exponential ease, cinematic on turns
const _chasePos = new THREE.Vector3();
const _chaseDir = new THREE.Vector3();
function updateChaseCam() {
  if (!chaseCam) return;
  const qpos = data.qpos;
  const yaw = Math.atan2(
    2 * (qpos[3] * qpos[6] + qpos[4] * qpos[5]),
    1 - 2 * (qpos[5] * qpos[5] + qpos[6] * qpos[6]),
  );
  const dist = camera.position.distanceTo(controls.target);
  const horiz = dist * Math.cos(CHASE_PITCH);
  const vert = dist * Math.sin(CHASE_PITCH);
  // Duck forward in MJCF is (cos yaw, sin yaw, 0); Z-up -> Y-up maps it
  // to three-space (cos yaw, 0, -sin yaw). Behind = minus that.
  _chasePos.set(
    controls.target.x - Math.cos(yaw) * horiz,
    controls.target.y + vert,
    controls.target.z + Math.sin(yaw) * horiz,
  );
  camera.position.lerp(_chasePos, CHASE_EASE);
  // Re-project onto the orbit sphere: lerping between two points at the
  // same radius cuts the chord, which would slowly zoom the camera in
  // during large swings (e.g. re-attaching after the duck turned around).
  _chaseDir.copy(camera.position).sub(controls.target);
  const len = _chaseDir.length();
  if (len > 1e-6) camera.position.copy(controls.target).addScaledVector(_chaseDir, dist / len);
  camera.lookAt(controls.target);
}
renderer.domElement.addEventListener("pointerdown", () => { chaseCam = false; });

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
  // Ball: raw MJCF pose inside the Z-up group, hidden while parked.
  ballMesh.visible = ballActive;
  if (ballActive) {
    ballMesh.position.set(qpos[ballQposAdr], qpos[ballQposAdr + 1], qpos[ballQposAdr + 2]);
    ballMesh.quaternion.set(
      qpos[ballQposAdr + 4], qpos[ballQposAdr + 5], qpos[ballQposAdr + 6], qpos[ballQposAdr + 3],
    );
  }
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
// Voice banks from the robot runtime: each printed duck ships a different
// voice, so each colourway gets its own bank and every quack draws a
// random chirp take from it - same as the real ducks all sounding
// slightly different. Audio elements are created lazily and cached.
// Browsers block audio until the first user gesture; the rejected play()
// is swallowed and sound simply starts working after the first click/key.
const CHIRP_TAKES = "abcdefghijkl";
const VOICE_BANK = { classic: "duck1", charcoal: "duck2", purple: "duck3", blue: "duck4" };
const chirpCache = new Map();
function playChirp() {
  const bank = VOICE_BANK[currentVariant] ?? "duck1";
  const take = CHIRP_TAKES[(Math.random() * CHIRP_TAKES.length) | 0];
  const url = signed(`./assets/voices/${bank}/chirp_${take}.wav`);
  let a = chirpCache.get(url);
  if (!a) {
    a = new Audio(url);
    a.volume = 0.7;
    chirpCache.set(url, a);
  }
  a.currentTime = 0;
  a.play().catch(() => {});
}
// Silent jaw flap for mode/colour changes; the actual quack sound only
// plays on the gamepad right trigger (anything more gets noisy fast).
const quack = () => {
  quackAt = performance.now();
};
const quackLoud = () => {
  quack();
  playChirp();
};
function jawOpenNow() {
  const t = (performance.now() - quackAt) / QUACK_MS;
  const flap = t >= 0 && t < 1 ? Math.sin(Math.PI * t) : 0;
  return Math.max(flap, padJaw);
}
function syncJaw() {
  setJawOpen(rig, jawOpenNow());
}

// HUD elements for the mini command sticks + hint highlighting.
const el = (id) => document.getElementById(id);
const dotMove = el("dot-move"), dotTurn = el("dot-turn");
const boxMove = dotMove.parentElement, boxTurn = dotTurn.parentElement;
const keyEls = {
  fwd: el("key-fwd"), back: el("key-back"),
  turnl: el("key-turnl"), turnr: el("key-turnr"),
  left: el("key-left"), right: el("key-right"),
    roll: el("key-roll"), reset: el("key-reset"), kick: el("key-kick"),
    ball: el("key-ball"), cam: el("key-cam"),
    padX: el("key-pad-x"), padSit: el("key-pad-sit"),
    padRun: el("key-pad-run"), padRt: el("key-pad-rt"),
    padRb: el("key-pad-rb"), padLb: el("key-pad-lb"),
    padY: el("key-pad-y"),
  };
const STICK_R = 15; // px, max dot travel inside the 46px stick circle
let resetFlashAt = -Infinity;
let ballFlashAt = -Infinity;
let padYFlashAt = -Infinity;

function renderStats() {
  const [vx, vy, wz] = effectiveCmd();
  // The active policy lives in the big center label and the twist in the
  // mini sticks; up here only the bare telemetry remains.
  const peers = ghosts?.peerCount() ?? 0;
  statsEl.textContent =
    `ctrl ${ctrlHz.toFixed(0)} Hz \u00b7 sim t ${data.time.toFixed(1)} s` +
    (peers ? ` \u00b7 ${peers + 1} online` : "");

  // Mini sticks: the dot mirrors the effective twist, lit yellow while the
  // user is actually driving.
  const manual = (padActive || held.size > 0) && mode !== "roll";
  const yN = vx >= 0 ? vx / VEL_FWD : vx / -VEL_BACK;
  dotMove.style.transform = `translate(${(-vy / VEL_LAT) * STICK_R}px, ${-yN * STICK_R}px)`;
  dotTurn.style.transform = `translate(${(-wz / VEL_ANG) * STICK_R}px, 0px)`;
  boxMove.classList.toggle("live", manual && (Math.abs(vx) > 0.01 || Math.abs(vy) > 0.01));
  boxTurn.classList.toggle("live", manual && Math.abs(wz) > 0.01);

  // Keycap highlighting: each individual key lights only while its own
  // action is active, and only for its own input device.
  const sitting = mode === "sitstand" && sitFlag === 1;
  for (const k of ["fwd", "back", "turnl", "turnr", "left", "right"]) {
    keyEls[k].classList.toggle("lit", held.has(k));
  }
  keyEls.roll.classList.toggle("lit", mode === "roll" && rollSource === "kb");
  keyEls.kick.classList.toggle("lit", isKick() && kickSource === "kb");
  keyEls.reset.classList.toggle("lit", performance.now() - resetFlashAt < 400);
  keyEls.ball.classList.toggle("lit", performance.now() - ballFlashAt < 400);
  keyEls.cam.classList.toggle("lit", chaseCam); // steady while chasing
  keyEls.padX.classList.toggle("lit", mode === "roll" && rollSource === "pad");
  keyEls.padY.classList.toggle("lit", performance.now() - padYFlashAt < 400);
  keyEls.padRb.classList.toggle("lit", mode === "kickR" && kickSource === "pad");
  keyEls.padLb.classList.toggle("lit", mode === "kickL" && kickSource === "pad");
  keyEls.padSit.classList.toggle("lit", sitting);
  keyEls.padRun.classList.toggle("lit", padPrev.dpadUp);
  keyEls.padRt.classList.toggle("lit", padJaw > 0.3);
}

// Real implementation assigned in the gamepad section below; the render
// loop starts before that section has evaluated, hence the indirection.
let pollPad = () => {};
// Multiplayer ghosts, initialised asynchronously at the end of the module.
let ghosts = null;

function loop() {
  requestAnimationFrame(loop);
  pollPad();
  syncRig();
  syncJaw();
  ghosts?.update();
  controls.update();
  updateChaseCam();
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

// e.code is physical position, so one map covers QWERTY and AZERTY:
// WASD/ZQSD run + turn, the physical Q/E row (A/E on AZERTY) strafes.
const KEYMAP = {
  ArrowUp: "fwd", KeyW: "fwd",
  ArrowDown: "back", KeyS: "back",
  ArrowLeft: "turnl", KeyA: "turnl",
  ArrowRight: "turnr", KeyD: "turnr",
  KeyQ: "left", KeyE: "right",
};

// Keyboard kicks alternate feet (the pad picks explicitly via RB/LB).
let kickNextLeft = false;
window.addEventListener("keydown", (e) => {
  if (e.repeat) return;
  if (e.code === "KeyR") { resetSim(); resetFlashAt = performance.now(); return; }
  if (e.code === "KeyB") { spawnBall(); ballFlashAt = performance.now(); return; }
  if (e.code === "KeyC") { chaseCam = !chaseCam; return; }
  if (e.code === "Space") { e.preventDefault(); triggerRoll(); return; }
  if (e.code === "KeyF") {
    e.preventDefault();
    if (triggerKick(kickNextLeft ? "left" : "right")) kickNextLeft = !kickNextLeft;
    return;
  }
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
// X = roll, DpadDown = sit/stand, DpadUp = back to run, RT = mouth.
const PAD_DEADZONE = 0.15;
const PAD_ALPHA = 0.12;
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
  // (then the keyboard takes over again).
  const stickInput = lx !== 0 || ly !== 0 || rx !== 0;
  if (stickInput) padActive = true;
  else if (padActive && Math.abs(padCmd[0]) + Math.abs(padCmd[1]) + Math.abs(padCmd[2]) < 0.01) {
    padActive = false;
    padCmd.fill(0);
  }

  // Standard mapping indices: X=2, Y=3, LB=4, RB=5, DpadDown=13, RT=7 (analog).
  const x = !!gp.buttons[2]?.pressed;
  if (x && !padPrev.x) triggerRoll("pad");
  padPrev.x = x;

  // Y: pop / respawn the ball, same rising-edge pattern as X.
  const y = !!gp.buttons[3]?.pressed;
  if (y && !padPrev.y) { spawnBall(); padYFlashAt = performance.now(); }
  padPrev.y = y;

  // RB / LB: right / left kick, same buttons as the robot runtime.
  const rb = !!gp.buttons[5]?.pressed;
  if (rb && !padPrev.rb) triggerKick("right", "pad");
  padPrev.rb = rb;
  const lb = !!gp.buttons[4]?.pressed;
  if (lb && !padPrev.lb) triggerKick("left", "pad");
  padPrev.lb = lb;

  const dpadDown = !!gp.buttons[13]?.pressed;
  if (dpadDown && !padPrev.dpadDown) {
    const sitting = mode === "sitstand" && sitFlag === 1;
    setMode(sitting ? "walk" : "sit");
  }
  padPrev.dpadDown = dpadDown;

  // DpadUp: straight back to running. Ignored mid-roll (it hands back
  // to walk on its own, and switching on a tipped duck would floor it).
  const dpadUp = !!gp.buttons[12]?.pressed;
  if (dpadUp && !padPrev.dpadUp && mode !== "walk" && mode !== "roll") setMode("walk");
  padPrev.dpadUp = dpadUp;

  // Triggers drive the mouth like the runtime (max of both), and the right
  // one quacks on its rising edge — same as the robot's chirp.
  // Schmitt trigger on the analog RT value: fire at 0.35, re-arm only once
  // it drops below 0.2. A single threshold re-fires on every jitter around
  // it, which is how one squeeze used to quack several times.
  const rt = gp.buttons[7]?.value ?? 0;
  const lt = gp.buttons[6]?.value ?? 0;
  padJaw = Math.max(rt, lt);
  if (padPrev.rtArmed && rt >= 0.35) { quackLoud(); padPrev.rtArmed = false; }
  else if (!padPrev.rtArmed && rt < 0.2) padPrev.rtArmed = true;
};

// Read-only state label (bottom-left): reflects the active policy,
// switching happens via keyboard/gamepad only.
const modeLabel = document.getElementById("mode-label");

function setMode(next) {
  // No policy switching mid-roll or mid-kick: both end on their own and
  // return to walk - switching now would floor the duck.
  if ((mode === "roll" && rollRun) || (isKick() && kickRun)) return;
  clearModeTimers();
  quack();
  rollRun = null;
  if (next !== "sit") {
    // Leaving a sit: let the sitstand policy stand the duck back up first.
    if (mode === "sitstand" && sitFlag === 1) {
      sitFlag = 0;
      standTimer = setTimeout(() => {
        standTimer = null;
        mode = next;
        lastAction.fill(0);
        syncButtons();
      }, 2000);
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
    sitTimer = setTimeout(() => {
      sitTimer = null;
      if (mode === "sitstand") { sitFlag = 1; syncButtons(); }
    }, 800);
  }
  syncButtons();
}

// One roll, then straight back to running (Space key or the button).
// lastAction is deliberately NOT zeroed here: the robot runtime keeps one
// continuous action history across policy switches, and the roll initiates
// more reliably mid-gait with the true last actions in the obs.
function triggerRoll(source = "kb") {
  // Rolls only launch from a standing walk: from a sit (or mid sit/stand
  // hand-over) the roll policy just faceplants the duck.
  if (mode !== "walk" || standTimer) return;
  clearModeTimers();
  rollSource = source;
  quack();
  mode = "roll";
  sitFlag = 0;
  rollRun = { steps: 0, tipped: false };
  syncButtons();
}

// One blind kick (the duck can't see any ball - it's a scripted boot),
// left or right leg. Same launch constraints as the roll. Returns whether
// the kick actually launched so the keyboard's foot alternation only
// advances on real kicks.
function triggerKick(foot, source = "kb") {
  if (mode !== "walk" || standTimer) return false;
  clearModeTimers();
  kickSource = source;
  quack();
  mode = foot === "left" ? "kickL" : "kickR";
  sitFlag = 0;
  kickRun = { steps: 0 };
  syncButtons();
  return true;
}
// Slot-machine roll: the old text slides up and out while the new one
// rises in from below the (overflow-hidden) label box.
function setModeLabel(text) {
  const cur = modeLabel.lastElementChild;
  if (cur && cur.textContent === text) return;
  const next = document.createElement("span");
  next.className = "mode-text in";
  next.textContent = text;
  modeLabel.appendChild(next);
  void next.offsetWidth; // flush layout so the transition actually plays
  if (cur) {
    cur.classList.add("out");
    setTimeout(() => cur.remove(), 350);
  }
  next.classList.remove("in");
}

function syncButtons() {
  const sitting = mode === "sitstand" && sitFlag === 1;
  setModeLabel(mode === "roll" ? "Roll" : isKick() ? "Kick" : sitting ? "Sit" : "Run");
}

// ── Colour swatches: re-skin the rig live, with a quack ─────────────────
  // One representative colour per variant so the dots read at a glance.
// Variants can force theirs with a `swatch` spec (purple does: its head
// is warm gray but its identity is the purple accents).
const SWATCH_SLOT = { classic: "feet", charcoal: "headDome", purple: "feet", blue: "facePlate" };
const swatchesEl = document.getElementById("swatches");
const swatchBtns = new Map();
for (const name of VARIANT_NAMES) {
  const v = VARIANTS[name];
  const b = document.createElement("button");
  b.style.background = specToHex(v.swatch ?? v[SWATCH_SLOT[name] ?? "bodyShell"]);
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
  model, data, mujoco, camera, controls,
  get mode() { return mode; },
  get sitFlag() { return sitFlag; },
  buildObs, cmd,
  velCmd, lastAction, resetSim,
  spawnBall,
  get ballActive() { return ballActive; },
  get ballQposAdr() { return ballQposAdr; },
  get chaseCam() { return chaseCam; },
  set chaseCam(v) { chaseCam = !!v; },
  step: async (n = 1) => { for (let i = 0; i < n; i++) await controlStep(); },
  render: () => { syncRig(); renderer.render(scene, camera); },
  // One full render-loop iteration, for tests driving frames manually.
  frame: () => { syncRig(); syncJaw(); controls.update(); updateChaseCam(); renderStats(); renderer.render(scene, camera); },
  get ghosts() { return ghosts; },
};

// ── Multiplayer ghosts (WebRTC, serverless signaling) ───────────────────
// Broadcast this duck's pose and render up to 3 other visitors live as
// translucent ducks. Fire-and-forget: any failure just means no ghosts.
const r3 = (x) => Math.round(x * 1000) / 1000;
try {
  const { initGhosts } = await import(signed(`./ghosts.js?v=${SELF_V}`));
  ghosts = await initGhosts({
    scene, rig, cloneRig, setJoint, setJawOpen, applyVariant,
    jointNames: JOINT_NAMES,
    getLocalState: () => {
      const qpos = data.qpos;
      const j = new Array(NUM_JOINTS);
      for (let i = 0; i < NUM_JOINTS; i++) j[i] = r3(qpos[qposAdr[i]]);
      return {
        p: [r3(qpos[0]), r3(qpos[1]), r3(qpos[2]), r3(qpos[3]), r3(qpos[4]), r3(qpos[5]), r3(qpos[6])],
        j,
        w: r3(jawOpenNow()),
        v: currentVariant,
      };
    },
  });
} catch (e) {
  window.__ghostErr = String((e && e.stack) || e);
  console.warn("ghosts disabled:", e);
}
