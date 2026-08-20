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
// No strafe input anymore: the lateral cmd slot stays zeroed for the obs.
const VEL_FWD = 0.25, VEL_BACK = -0.2, VEL_ANG = 1.0;

// Kickable ball: radius and parking spot (far away = hidden by default).
const BALL_RADIUS = 0.05;
const BALL_PARK_POS = "50 0 0.05";

// Square arena boxing the play area: static walls at +-ARENA_HALF keep
// the ball (and the duck) inside. Tall enough that neither steps over.
const ARENA_HALF = 1.5; // inner half-size, m
const ARENA_WALL_H = 0.25;
const ARENA_WALL_T = 0.05;

const mount = document.getElementById("scene");
const loadingEl = document.getElementById("loading");
const hudEl = document.getElementById("hud");
const statsEl = document.getElementById("stats");
const osdTimeEl = document.getElementById("osd-time");

// BIOS/POST boot readout. The real load runs silently behind the welcome
// modal and only RECORDS milestones into bootLog; the readout itself plays
// after the first "Waddle in": a rapid-fire replay when everything already
// finished, or an honest live tracker (cursor blinking on the pending
// line) when the user enters mid-load. Never slows the actual boot.
const postEl = loadingEl.querySelector(".post");
const bootLog = []; // { label, status, raw, el } - status null = pending
const lineText = (e) =>
  e.raw || e.status === null ? e.label : `${e.label} `.padEnd(26, ".") + ` ${e.status}`;
const bootLine = (label) => {
  const entry = { label, status: null, raw: false, el: null };
  bootLog.push(entry);
  return (status = "OK") => {
    entry.status = status;
    if (entry.el) entry.el.textContent = lineText(entry);
  };
};
// Plain line with no dotted leader (header / error text).
const bootNote = (label) => {
  bootLog.push({ label, status: "", raw: true, el: null });
};
bootNote("Microduck BIOS v1.0");
bootLine("MEMORY CHECK")("640K OK");
bootLine("DUCK FIRMWARE")("PRESENT");

let bootDone = false;
let biosStarted = false;
const biosSleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function playBios() {
  if (biosStarted) return;
  biosStarted = true;
  loadingEl.style.display = "flex";
  let i = 0;
  for (;;) {
    if (i < bootLog.length) {
      const entry = bootLog[i++];
      entry.el = document.createElement("div");
      entry.el.textContent = lineText(entry);
      postEl.appendChild(entry.el);
      // Honest mode: hold on a stage still in flight (its completer will
      // fill in the status), cursor blinking on it via CSS.
      while (entry.status === null && !bootDone) await biosSleep(60);
      await biosSleep(105); // rapid-fire pacing for finished lines
    } else if (bootDone) {
      break;
    } else {
      await biosSleep(60); // waiting for the next real stage
    }
  }
  const ready = document.createElement("div");
  ready.textContent = "READY.";
  postEl.appendChild(ready);
  await biosSleep(500);
  loadingEl.classList.add("off");
  await biosSleep(500);
  loadingEl.style.display = "none";
}
// Only the initial entry triggers the replay; reopening the modal from
// the brand later never replays (biosStarted latches).
if (window.__microduckEntered) playBios();
else document.addEventListener("microduck:enter", () => playBios(), { once: true });

// Surface boot failures on the page itself: a rejected top-level await
// otherwise leaves a dead page with no visible error.
window.addEventListener("unhandledrejection", (e) => {
  const msg = e.reason?.stack || e.reason?.message || String(e.reason);
  bootNote(`Boot failed: ${msg}`);
  playBios();
  console.error("[rl] unhandled rejection", e.reason);
});
window.addEventListener("error", (e) => {
  bootNote(`Boot failed: ${e.message}`);
  playBios();
});
const traced = (label, p) => {
  const done = bootLine(label);
  return p.then(
    (v) => { done("OK"); return v; },
    (err) => { done("FAIL"); console.error(`[rl] ${label} FAILED`, err); throw err; },
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
  // Arena walls: four static boxes (no joints, so no qpos/keyframe
  // impact); default contype/conaffinity collides with ball and duck.
  const ht = ARENA_WALL_T / 2, hh = ARENA_WALL_H / 2;
  const off = ARENA_HALF + ht, span = ARENA_HALF + ARENA_WALL_T;
  const walls = [
    { name: "wall_px", pos: `${off} 0 ${hh}`, size: `${ht} ${span} ${hh}` },
    { name: "wall_nx", pos: `${-off} 0 ${hh}`, size: `${ht} ${span} ${hh}` },
    { name: "wall_py", pos: `0 ${off} ${hh}`, size: `${span} ${ht} ${hh}` },
    { name: "wall_ny", pos: `0 ${-off} ${hh}`, size: `${span} ${ht} ${hh}` },
  ];
  for (const w of walls) {
    doc.querySelector("worldbody").appendChild(
      el("geom", { name: w.name, type: "box", pos: w.pos, size: w.size }),
    );
  }
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
const [mujoco, { xml, meshFiles }, k] = await Promise.all([
  traced("MUJOCO WASM", loadMujoco()),
  traced("PHYSICS MJCF", buildPhysicsXml()),
  traced("KINEMATICS", loadKinematics(`${MODEL_DIR}/kinematics.json`)),
]);

const doneMeshes = bootLine("MESH ASSETS");
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
doneMeshes(`${meshFiles.length} FILES`);

const sessions = {};
let currentVariant = randomVariantName();
const rigPromise = (async () => {
  const doneRig = bootLine("RENDER RIG");
  const rig = await buildRig(k, { materialForMesh: materialHookFor(VARIANTS[currentVariant]) });
  doneRig("OK");
  return rig;
})();
const donePolicies = bootLine("LOADING POLICIES");
const sessionOpts = { executionProviders: ["wasm"] };
[sessions.walk, sessions.sitstand, sessions.roll, sessions.kickL, sessions.kickR] =
  await Promise.all([
    ort.InferenceSession.create(signed(POLICIES.walk), sessionOpts),
    ort.InferenceSession.create(signed(POLICIES.sitstand), sessionOpts),
    ort.InferenceSession.create(signed(POLICIES.roll), sessionOpts),
    ort.InferenceSession.create(signed(POLICIES.kickL), sessionOpts),
    ort.InferenceSession.create(signed(POLICIES.kickR), sessionOpts),
  ]);
donePolicies("5/5");

const doneCompile = bootLine("COMPILING PHYSICS");
const model = mujoco.MjModel.from_xml_string(xml, vfs);
const data = new mujoco.MjData(model);
doneCompile("COMPILED");

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
const padPrev = { x: false, y: false, rb: false, lb: false, r3: false, dpadDown: false, dpadUp: false, rtArmed: true };
// Right-stick camera state, read by renderStats before the gamepad
// section below has evaluated (same reason as padCmd above).
let padOrbitLive = false; // HUD: RS keycap lit while deflected
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
  if (mode === "roll" || isKick() || postKickLock > 0) return ZERO_CMD;
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
let KICK_STEPS = 25;
// Post-kick grace: keep commands zeroed for a beat after the kick window
// hands back to walk, so the duck finishes the strike cleanly instead of
// instantly sprinting off. Step-counted like everything else.
const POST_KICK_LOCK_STEPS = 20; // 0.4 s at 50 Hz
let postKickLock = 0;

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
  postKickLock = 0;
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
  // Clamp inside the arena: a duck standing against a wall must not pop
  // the ball into (or beyond) it.
  const lim = ARENA_HALF - BALL_RADIUS - 0.05;
  const clamp = (v) => Math.min(lim, Math.max(-lim, v));
  qpos[ballQposAdr] = clamp(qpos[0] + Math.cos(heading) * dist);
  qpos[ballQposAdr + 1] = clamp(qpos[1] + Math.sin(heading) * dist);
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

  // Ball respawn watchdog: with the arena walls the ball can no longer
  // legitimately leave, so this is a safety net for solver tunnelling -
  // outside the arena bounds means "escaped through a glitch", bring it
  // back near the duck. Re-read qpos: resetSim above may have re-parked
  // the ball.
  if (ballActive) {
    const q = data.qpos;
    const escaped =
      Math.abs(q[ballQposAdr]) > ARENA_HALF + 0.1 ||
      Math.abs(q[ballQposAdr + 1]) > ARENA_HALF + 0.1;
    if (escaped) spawnBall();
  }

  // One-shot roll, step-counted like the robot runtime (a single roll is
  // ~1 s = 50 control steps there): hand back to walking once the trunk has
  // tipped over and is upright again, or after a hard 2 s window if the roll
  // never initiated. Counting steps instead of wall time keeps the logic
  // correct when the sim is fast-forwarded or the tab is throttled.
  // One-shot kick: fixed 0.5 s window like the robot runtime, then straight
  // back to walking. lastAction is NOT zeroed on either swap - the runtime
  // keeps one continuous action history across policy switches.
  if (postKickLock > 0 && mode === "walk") postKickLock--;

  if (isKick() && kickRun) {
    kickRun.steps++;
    if (kickRun.steps >= KICK_STEPS) {
      kickRun = null;
      mode = "walk";
      postKickLock = POST_KICK_LOCK_STEPS;
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
      uCellColor: { value: new THREE.Color(0x8e8371) },
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
      // Tron-style line: a thicker antialiased core plus a faint, much
      // wider halo added on top (squared falloff keeps it a whisper of a
      // glow rather than a bloom wash).
      float gridLine(vec2 p, float size) {
        vec2 r = p / size;
        vec2 g = abs(fract(r - 0.5) - 0.5) / fwidth(r);
        float d = min(g.x, g.y);
        float core = 1.0 - smoothstep(0.0, 1.8, d);
        float halo = 1.0 - smoothstep(0.0, 7.0, d);
        return core + halo * halo * 0.22;
      }
      void main() {
        float cell = gridLine(vWorld.xz, uCell);
        float section = gridLine(vWorld.xz, uSection);
        float d = distance(vWorld.xz, uFocus.xz);
        float fade = pow(clamp(1.0 - d / uFadeDist, 0.0, 1.0), 1.6);
        vec3 col = mix(uCellColor, uSectionColor, clamp(section, 0.0, 1.0));
        float alpha = min(max(section * 0.6, cell * 0.4) * fade, 1.0);
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

// Arena walls, drawn in the same grid language as the floor: identical
// cell/section lines from world coordinates, same radial fade around the
// duck, plus a vertical fade toward the top edge so the walls read as a
// light enclosure instead of solid slabs.
function makeWallGridMaterial(alongX) {
  return new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    uniforms: {
      uCell: { value: 0.1 },
      uSection: { value: 0.5 },
      uCellColor: { value: new THREE.Color(0x8e8371) },
      uSectionColor: { value: new THREE.Color(0xffb366) },
      // Gentler radial fade than the floor: the walls sit 1.5+ m from the
      // duck by construction and would vanish with the floor's 3 m fade.
      uFadeDist: { value: 5.0 },
      uFocus: { value: new THREE.Vector3() },
      uWallH: { value: ARENA_WALL_H },
      uAlongX: { value: alongX ? 1.0 : 0.0 },
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
      uniform float uCell, uSection, uFadeDist, uWallH, uAlongX;
      uniform vec3 uCellColor, uSectionColor, uFocus;
      // Same Tron-style core + faint halo as the floor grid.
      float gridLine(vec2 p, float size) {
        vec2 r = p / size;
        vec2 g = abs(fract(r - 0.5) - 0.5) / fwidth(r);
        float d = min(g.x, g.y);
        float core = 1.0 - smoothstep(0.0, 1.8, d);
        float halo = 1.0 - smoothstep(0.0, 7.0, d);
        return core + halo * halo * 0.22;
      }
      void main() {
        // Wall surface coords: the in-plane horizontal world axis + height.
        float h = mix(vWorld.z, vWorld.x, uAlongX);
        vec2 p = vec2(h, vWorld.y);
        float cell = gridLine(p, uCell);
        float section = gridLine(p, uSection);
        float d = distance(vWorld.xz, uFocus.xz);
        float fade = pow(clamp(1.0 - d / uFadeDist, 0.0, 1.0), 1.6);
        float vert = 1.0 - clamp(vWorld.y / uWallH, 0.0, 1.0);
        vec3 col = mix(uCellColor, uSectionColor, clamp(section, 0.0, 1.0));
        float alpha = min(max(section * 0.9, cell * 0.6) * fade * (0.3 + 0.7 * vert), 1.0);
        if (alpha < 0.004) discard;
        gl_FragColor = vec4(col, alpha);
      }
    `,
  });
}
const wallMats = [];
{
  const wallLen = 2 * (ARENA_HALF + ARENA_WALL_T);
  // (three coords: MJCF x -> x, MJCF y -> -z; walls sit at their inner faces)
  const wallDefs = [
    { x: ARENA_HALF, z: 0, rotY: -Math.PI / 2, alongX: false },
    { x: -ARENA_HALF, z: 0, rotY: Math.PI / 2, alongX: false },
    { x: 0, z: ARENA_HALF, rotY: Math.PI, alongX: true },
    { x: 0, z: -ARENA_HALF, rotY: 0, alongX: true },
  ];
  for (const w of wallDefs) {
    const mat = makeWallGridMaterial(w.alongX);
    wallMats.push(mat);
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(wallLen, ARENA_WALL_H), mat);
    mesh.position.set(w.x, ARENA_WALL_H / 2, w.z);
    mesh.rotation.y = w.rotY;
    scene.add(mesh);
  }
}

const rig = await rigPromise;
scene.add(rig.placer);
const trunkGroup = rig.bodies.get("trunk_base");

// Soccer-ball look computed per pixel on the sphere itself, so there is
// no pole or seam special case by design. The truncated icosahedron is
// reconstructed as a spherical Voronoi diagram over 32 sites: the 12
// icosahedron vertices (black pentagon centers, one sitting at each
// pole) and its 20 face centers (white hexagon centers). A pixel is
// black when its nearest site is a pentagon center and it sits clear of
// the cell boundary by a seam margin - which yields big flat-edged black
// pentagons separated from the white hexagons by thin seams, corners
// almost touching, exactly like the real panel layout.
function makeSoccerBallTexture() {
  const W = 1024, H = 512;
  const c = document.createElement("canvas");
  c.width = W;
  c.height = H;
  const ctx = c.getContext("2d");
  // 12 icosahedron vertices: 2 poles + two staggered rings of 5 at
  // latitude +-atan(1/2) (~26.57 deg) - the pentagon centers.
  const sites = [];
  const addSite = (v, isPent) => {
    const n = Math.hypot(v[0], v[1], v[2]);
    sites.push({ x: v[0] / n, y: v[1] / n, z: v[2] / n, pent: isPent });
  };
  const verts = [[0, 0, 1], [0, 0, -1]];
  const latR = Math.atan(0.5), cr = Math.cos(latR), sr = Math.sin(latR);
  for (let i = 0; i < 5; i++) {
    const a = (i * 72 * Math.PI) / 180;
    const b = ((i * 72 + 36) * Math.PI) / 180;
    verts.push([cr * Math.cos(a), cr * Math.sin(a), sr]);
    verts.push([cr * Math.cos(b), cr * Math.sin(b), -sr]);
  }
  for (const v of verts) addSite(v, true);
  // 20 face centers (hexagon centers): normalized centroids of every
  // mutually-adjacent vertex triple (adjacent pairs have dot = 1/sqrt(5)).
  const adj = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2] > 0.3;
  for (let i = 0; i < 12; i++) {
    for (let j = i + 1; j < 12; j++) {
      if (!adj(verts[i], verts[j])) continue;
      for (let k = j + 1; k < 12; k++) {
        if (adj(verts[i], verts[k]) && adj(verts[j], verts[k])) {
          addSite([
            verts[i][0] + verts[j][0] + verts[k][0],
            verts[i][1] + verts[j][1] + verts[k][1],
            verts[i][2] + verts[j][2] + verts[k][2],
          ], false);
        }
      }
    }
  }
  // Seam half-width and anti-alias band, in radians of arc.
  const SEAM = (1.6 * Math.PI) / 180;
  const AA = (0.35 * Math.PI) / 180;
  // Groove reach for the bump map: a touch wider than the painted seam so
  // the recess shoulders catch light on both sides of the line.
  const GROOVE = SEAM * 1.5;
  const BG = [233, 231, 224], INK = [23, 23, 29], STITCH = [200, 197, 188];
  const img = ctx.createImageData(W, H);
  const px = img.data;
  // Height map sharing the same panel construction: seams become recessed
  // grooves, plus a very fine leather/PVC grain over the whole surface.
  const bc = document.createElement("canvas");
  bc.width = W;
  bc.height = H;
  const bctx = bc.getContext("2d");
  const bimg = bctx.createImageData(W, H);
  const bpx = bimg.data;
  for (let row = 0; row < H; row++) {
    const lat = Math.PI / 2 - ((row + 0.5) / H) * Math.PI;
    const cl = Math.cos(lat), sl = Math.sin(lat);
    for (let col = 0; col < W; col++) {
      const lon = ((col + 0.5) / W) * 2 * Math.PI - Math.PI;
      const dx = cl * Math.cos(lon), dy = cl * Math.sin(lon), dz = sl;
      let best = -2, second = -2, bestPent = false;
      for (const s of sites) {
        const d = dx * s.x + dy * s.y + dz * s.z;
        if (d > best) { second = best; best = d; bestPent = s.pent; }
        else if (d > second) second = d;
      }
      // Signed distance to the Voronoi cell boundary along the geodesic.
      const halfGap = (Math.acos(Math.min(1, second)) - Math.acos(Math.min(1, best))) / 2;
      // Black panel: inside a pentagon cell, clear of the seam margin.
      const black = bestPent ? Math.min(1, Math.max(0, (halfGap - SEAM) / AA)) : 0;
      // Subtle stitch line on every remaining cell boundary so the white
      // hexagons read as panels too.
      const stitch = Math.min(1, Math.max(0, 1 - halfGap / (SEAM * 0.6))) * (1 - black);
      const o = (row * W + col) * 4;
      for (let ch = 0; ch < 3; ch++) {
        const base = BG[ch] + (STITCH[ch] - BG[ch]) * stitch;
        px[o + ch] = base + (INK[ch] - base) * black;
      }
      px[o + 3] = 255;
      // Bump: quadratic groove profile (soft shoulders, no golf-ball
      // embossing) + grain noise.
      const groove = Math.max(0, 1 - halfGap / GROOVE) ** 2;
      const hgt = 205 - groove * 115 + (Math.random() - 0.5) * 14;
      const h8 = Math.max(0, Math.min(255, hgt));
      bpx[o] = h8; bpx[o + 1] = h8; bpx[o + 2] = h8;
      bpx[o + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  bctx.putImageData(bimg, 0, 0);
  const finish = (canvas, srgb) => {
    const tex = new THREE.CanvasTexture(canvas);
    // The bump map stays linear; only the color map is sRGB.
    if (srgb) tex.colorSpace = THREE.SRGBColorSpace;
    // Texel footprints get extremely anamorphic near the UV poles; without
    // anisotropy the cap edge visibly scallops at close range.
    tex.anisotropy = renderer.capabilities.getMaxAnisotropy();
    return tex;
  };
  return { map: finish(c, true), bumpMap: finish(bc, false) };
}
// Same Z-up -> Y-up trick as the duck rig: the group takes the axis fix,
// the mesh inside takes the raw MJCF free-joint pose.
const ballGroup = new THREE.Group();
ballGroup.rotation.x = -Math.PI / 2;
const ballTex = makeSoccerBallTexture();
const ballMesh = new THREE.Mesh(
  // 48x32 segments: the coarser default makes the UV interpolation near
  // the poles visibly scallop the round cap edge of the texture.
  new THREE.SphereGeometry(BALL_RADIUS, 48, 32),
  // Physical material for the waxed vintage-leather look: matte-ish base
  // with a whisper of clearcoat so highlights ride the seam grooves.
  new THREE.MeshPhysicalMaterial({
    map: ballTex.map,
    bumpMap: ballTex.bumpMap,
    bumpScale: 0.0012,
    metalness: 0,
    roughness: 0.55,
    clearcoat: 0.2,
    clearcoatRoughness: 0.35,
  }),
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
// During one-shot rolls and kicks the trunk tumbles, so the yaw read from
// its quaternion spins wildly and would whip the camera around. Lazily
// latch the last healthy yaw while walking/sitting and hold it for the
// whole one-shot; the position/target easing keeps following the trunk.
// No timers needed: both one-shots deterministically hand back to walk,
// and the camera position lerp absorbs the small heading correction when
// live tracking resumes (rolls end facing roughly the same way).
let chaseHeldYaw = 0;
function updateChaseCam() {
  if (!chaseCam) return;
  const qpos = data.qpos;
  let yaw;
  if (mode === "roll" || isKick()) {
    yaw = chaseHeldYaw;
  } else {
    yaw = Math.atan2(
      2 * (qpos[3] * qpos[6] + qpos[4] * qpos[5]),
      1 - 2 * (qpos[5] * qpos[5] + qpos[6] * qpos[6]),
    );
    chaseHeldYaw = yaw;
  }
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
  // Keep the grid plane (and its fade center) under the action; the wall
  // grids share the same radial fade focus.
  grid.position.set(controls.target.x, 0, controls.target.z);
  grid.material.uniforms.uFocus.value.copy(controls.target);
  for (const m of wallMats) m.uniforms.uFocus.value.copy(controls.target);
}

// Quack: jaw + chirp on the gamepad right trigger only. The jaw isn't a
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
// The jaw is driven ONLY by the dedicated gamepad triggers: the analog
// RT/LT value (padJaw) plus the flap that accompanies the RT-edge quack
// sound below. Mode changes, rolls, kicks and colour swaps no longer
// move the mouth (the old silent flap was removed by user request).
const quackLoud = () => {
  quackAt = performance.now();
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
    roll: el("key-roll"), reset: el("key-reset"),
    kickl: el("key-kickl"), kickr: el("key-kickr"),
    ball: el("key-ball"), cam: el("key-cam"),
    padX: el("key-pad-x"), padSit: el("key-pad-sit"),
    padRun: el("key-pad-run"), padRt: el("key-pad-rt"),
    padRb: el("key-pad-rb"), padLb: el("key-pad-lb"),
    padY: el("key-pad-y"), padRs: el("key-pad-rs"),
    padR3: el("key-pad-r3"),
  };
const STICK_R = 15; // px, max dot travel inside the 46px stick circle
let resetFlashAt = -Infinity;
let ballFlashAt = -Infinity;
let padYFlashAt = -Infinity;

function renderStats() {
  const [vx, , wz] = effectiveCmd();
  // The active policy lives in the big center label and the twist in the
  // mini sticks; up here only the bare telemetry remains.
  const peers = ghosts?.peerCount() ?? 0;
  // VHS counter: sim time as mm:ss:ff where ff counts control frames (50Hz).
  const t = data.time;
  const p2 = (n) => String(n).padStart(2, "0");
  osdTimeEl.textContent =
    `\u25b6 ${p2(Math.floor(t / 60))}:${p2(Math.floor(t) % 60)}:${p2(Math.floor((t % 1) * 50))}`;
  statsEl.textContent =
    `CTRL ${ctrlHz.toFixed(0)}HZ` + (peers ? ` \u00b7 ${peers + 1} ONLINE` : "");

  // Mini sticks: the dot mirrors the effective twist, lit yellow while the
  // user is actually driving.
  const manual = (padActive || held.size > 0) && mode !== "roll";
  const yN = vx >= 0 ? vx / VEL_FWD : vx / -VEL_BACK;
  // Move stick is vertical-only now that strafe is gone.
  dotMove.style.transform = `translate(0px, ${-yN * STICK_R}px)`;
  dotTurn.style.transform = `translate(${(-wz / VEL_ANG) * STICK_R}px, 0px)`;
  boxMove.classList.toggle("live", manual && Math.abs(vx) > 0.01);
  boxTurn.classList.toggle("live", manual && Math.abs(wz) > 0.01);

  // Keycap highlighting: each individual key lights only while its own
  // action is active, and only for its own input device.
  const sitting = mode === "sitstand" && sitFlag === 1;
  for (const k of ["fwd", "back", "turnl", "turnr"]) {
    keyEls[k].classList.toggle("lit", held.has(k));
  }
  keyEls.roll.classList.toggle("lit", mode === "roll" && rollSource === "kb");
  keyEls.kickl.classList.toggle("lit", mode === "kickL" && kickSource === "kb");
  keyEls.kickr.classList.toggle("lit", mode === "kickR" && kickSource === "kb");
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
  keyEls.padRs.classList.toggle("lit", padOrbitLive); // while deflected
  keyEls.padR3.classList.toggle("lit", chaseCam); // steady while chasing
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
// Boot complete: the sim/HUD go live immediately. The BIOS readout (if
// the user already waddled in, or when they do) sees bootDone and closes
// with READY. + fade on its own.
bootDone = true;
hudEl.hidden = false;
loop();

// ── Input: hold-to-command keys + HUD buttons ───────────────────────────
function refreshVelCmd() {
  velCmd[0] = held.has("fwd") ? VEL_FWD : held.has("back") ? VEL_BACK : 0;
  velCmd[2] = held.has("turnl") ? VEL_ANG : held.has("turnr") ? -VEL_ANG : 0;
}

// e.code is physical position, so one map covers QWERTY and AZERTY:
// arrows / WASD (ZQSD) run + turn. No strafe.
const KEYMAP = {
  ArrowUp: "fwd", KeyW: "fwd",
  ArrowDown: "back", KeyS: "back",
  ArrowLeft: "turnl", KeyA: "turnl",
  ArrowRight: "turnr", KeyD: "turnr",
};

window.addEventListener("keydown", (e) => {
  if (e.repeat) return;
  if (e.code === "Space") { e.preventDefault(); resetSim(); resetFlashAt = performance.now(); return; }
  if (e.code === "KeyB") { spawnBall(); ballFlashAt = performance.now(); return; }
  if (e.code === "KeyC") { chaseCam = !chaseCam; return; }
  if (e.code === "KeyR") { triggerRoll(); return; }
  // Physical Q/E (A/E on AZERTY): explicit left / right kick, mirroring
  // the pad's LB / RB.
  if (e.code === "KeyQ") { triggerKick("left"); return; }
  if (e.code === "KeyE") { triggerKick("right"); return; }
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
// Left stick: vertical = vx (asymmetric fwd/back), horizontal = turn,
// EMA-smoothed like the runtime's cmd_alpha. Right stick orbits the
// camera (detaches the chase cam, R3 re-toggles it). X = roll, Y = ball,
// RB/LB = kicks, DpadDown = sit/stand, DpadUp = back to run, RT = mouth.
const PAD_DEADZONE = 0.15;
const PAD_ALPHA = 0.12;
const dz = (v) => (Math.abs(v) < PAD_DEADZONE ? 0 : v);

// Right-stick camera orbit, in OrbitControls-compatible terms: rebuild
// the camera-target offset as a spherical, nudge azimuth/elevation, and
// re-place the camera. controls.update() then runs on the result, so the
// damping bookkeeping never fights it (same trick as updateChaseCam).
// The stick does not move the camera directly: it steers an angular
// VELOCITY that eases toward the stick's target rate (frame-rate
// independent exponential), so pushing ramps up gently and releasing
// coasts to a stop over ~0.3 s instead of freezing on the spot.
// Vertical is flight-style inverted by request: stick up orbits the
// camera downward, stick down orbits it upward.
const PAD_ORBIT_SPEED = 2.4; // rad/s at full deflection
const PAD_ORBIT_SMOOTH = 8; // 1/s response rate (~95% in 0.37 s)
const padOrbitVel = { az: 0, el: 0 }; // smoothed angular velocity, rad/s
const _padSph = new THREE.Spherical();
const _padOff = new THREE.Vector3();
function padOrbitStep(rx, ry, dt) {
  padOrbitLive = rx !== 0 || ry !== 0;
  if (padOrbitLive) chaseCam = false; // detach, same as a mouse grab
  const k = 1 - Math.exp(-PAD_ORBIT_SMOOTH * dt);
  padOrbitVel.az += (rx * PAD_ORBIT_SPEED - padOrbitVel.az) * k;
  // Inverted Y; elevation runs a touch slower, full-rate pitch is twitchy.
  padOrbitVel.el += (-ry * PAD_ORBIT_SPEED * 0.75 - padOrbitVel.el) * k;
  if (chaseCam) { padOrbitVel.az = 0; padOrbitVel.el = 0; return; }
  if (Math.abs(padOrbitVel.az) < 1e-3 && Math.abs(padOrbitVel.el) < 1e-3) return;
  _padOff.copy(camera.position).sub(controls.target);
  _padSph.setFromVector3(_padOff);
  // Stick right sweeps the camera right around the duck.
  _padSph.theta -= padOrbitVel.az * dt;
  _padSph.phi += padOrbitVel.el * dt;
  _padSph.phi = Math.min(controls.maxPolarAngle, Math.max(0.08, _padSph.phi));
  _padSph.makeSafe();
  camera.position.setFromSpherical(_padSph).add(controls.target);
  camera.lookAt(controls.target);
}

let padPollT = performance.now();
pollPad = function pollGamepad() {
  const now = performance.now();
  // Clamped so a background-tab stall can't slingshot the camera.
  const dt = Math.min((now - padPollT) / 1000, 0.05);
  padPollT = now;
  const gp = [...(navigator.getGamepads?.() ?? [])].find((p) => p && p.connected);
  document.body.classList.toggle("pad-connected", !!gp);
  if (!gp) {
    if (padActive) { padActive = false; padCmd.fill(0); padJaw = 0; }
    padOrbitLive = false;
    padOrbitVel.az = 0;
    padOrbitVel.el = 0;
    return;
  }
  // Left stick only: vertical = forward/back, horizontal = turn.
  // (No strafe; the right stick no longer drives movement.)
  const lx = dz(gp.axes[0] ?? 0), ly = dz(gp.axes[1] ?? 0);
  const up = -ly; // browser sticks report up as -1
  const target = [
    up >= 0 ? up * VEL_FWD : up * -VEL_BACK,
    0,
    -lx * VEL_ANG,
  ];
  for (let i = 0; i < 3; i++) padCmd[i] += PAD_ALPHA * (target[i] - padCmd[i]);
  // Sticks grab command authority on first input, release when back at rest
  // (then the keyboard takes over again).
  const stickInput = lx !== 0 || ly !== 0;
  if (stickInput) padActive = true;
  else if (padActive && Math.abs(padCmd[0]) + Math.abs(padCmd[1]) + Math.abs(padCmd[2]) < 0.01) {
    padActive = false;
    padCmd.fill(0);
  }

  // Right stick: camera orbit with velocity smoothing. Runs every frame
  // (not just while deflected) so a released stick coasts to a stop.
  padOrbitStep(dz(gp.axes[2] ?? 0), dz(gp.axes[3] ?? 0), dt);

  // R3 (right stick click): toggle the chase cam, gamepad twin of KeyC.
  const r3Btn = !!gp.buttons[11]?.pressed;
  if (r3Btn && !padPrev.r3) chaseCam = !chaseCam;
  padPrev.r3 = r3Btn;

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
  spawnBall, triggerKick, sessions, ort,
  get kickSteps() { return KICK_STEPS; },
  set kickSteps(v) { KICK_STEPS = v; },
  get ballActive() { return ballActive; },
  get ballQposAdr() { return ballQposAdr; },
  get chaseCam() { return chaseCam; },
  set chaseCam(v) { chaseCam = !!v; },
  padOrbitStep,
  jawOpenNow,
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
