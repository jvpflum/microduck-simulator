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
//
// Two locomotion variants share that exact interface (same 14 joints, same
// default pose, same 61D obs - verified from the ONNX metadata):
//   legs    - the walking robot (robot_allcollisions.xml), boot default
//   rollers - the skating variant (robot_allcollisions_rollers.xml): the
//             foot/sole assembly is replaced by a blade with 2 passive
//             wheels per leg (extra unactuated hinges in qpos, zero in the
//             keyframe). Lazy-loaded on the first M / DpadUp-hold switch.

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
const { VARIANTS, VARIANT_NAMES, materialHookFor, DEFAULT_VARIANT, applyVariant, specToHex } =
  await import(signed(`./variants.js?v=${SELF_V}`));
// Input stack: a video-game-style Controller aggregating pluggable sources
// (keyboard + gamepad today, touch later). controller.js documents the
// source interface contract and the action vocabulary.
const [{ Controller }, { KeyboardSource }, { GamepadSource }, { TouchSource }] = await Promise.all([
  import(signed(`./controls/controller.js?v=${SELF_V}`)),
  import(signed(`./controls/keyboard.js?v=${SELF_V}`)),
  import(signed(`./controls/gamepad.js?v=${SELF_V}`)),
  import(signed(`./controls/touch.js?v=${SELF_V}`)),
]);

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
  // Roller variant (lazy-loaded on first switch, never at boot):
  // drive = velocity-tracking skating, crouch = one-shot crouch-glide
  // driven by a phase encoding in the command slots (ground-pick style).
  drive: `${POLICY_DIR}/BEST_roller.onnx`,
  crouch: `${POLICY_DIR}/BEST_roller_crouch.onnx`,
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
// Roller mode limits, from the runtime's roller branch: asymmetric vx
// (0.6 push / 0.5 brake), no lateral. The real runtime launches rollers
// with --max-angular-vel 0.3: faster commanded turns tip the robot over,
// so the playground clamps wz the same way (keyboard and pad both go
// through velLims).
const RVEL_FWD = 0.6, RVEL_BACK = -0.5, RVEL_ANG = 0.3;
// Crouch-glide one-shot: command = [cos(2pi*phase), sin(2pi*phase), 0],
// phase advancing at 1/CROUCH_PERIOD_S per second and the cycle exiting
// at 0.7 - exactly the runtime's ground-pick slot the policy was trained
// against (mjlab CROUCH_PERIOD = 5.0, cycle end 0.7 => 3.5 s gesture).
const CROUCH_PERIOD_S = 5.0;
const CROUCH_END_PHASE = 0.7;

// Kickable ball: radius and parking spot (far away = hidden by default).
const BALL_RADIUS = 0.05;
const BALL_PARK_POS = "50 0 0.05";

// Square arena boxing the play area: static walls at +-ARENA_HALF keep
// the ball (and the duck) inside. Tall enough that neither steps over.
const ARENA_HALF = 1.5; // inner half-size, m
const ARENA_WALL_H = 0.25;
const ARENA_WALL_T = 0.05;
// Section grid: 5 cells across the 3 m arena (ODD, so a true middle
// column/row of cells exists; the lattice is shifted half a cell in the
// shaders so the walls land exactly on section lines).
const GRID_SECTION = (2 * ARENA_HALF) / 5; // 0.6 m
// Spawn: center of the middle section cell in the SECOND ROW FROM THE
// BACK wall. The duck faces +X (identity freejoint quat, walks toward
// local +X), so "back" is the -X wall: row centers sit at x = -1.2,
// -0.6, 0, 0.6, 1.2 -> second row is -0.6; middle column is y = 0.
// MJCF coordinates (three.js: x -> x, y -> -z).
const SPAWN_X = -ARENA_HALF + 1.5 * GRID_SECTION; // -0.6
const SPAWN_Y = 0;

const mount = document.getElementById("scene");
const loadingEl = document.getElementById("loading");
const hudEl = document.getElementById("hud");
const ctrlHzEl = document.getElementById("ctrl-hz");
const osdTimeEl = document.getElementById("osd-time");

// BIOS/POST boot readout. The real load runs silently behind the welcome
// modal and only RECORDS milestones into bootLog; the readout itself plays
// after the first "Waddle in": a rapid-fire replay when everything already
// finished, or an honest live tracker (cursor blinking on the pending
// line) when the user enters mid-load. Never slows the actual boot.
const postEl = loadingEl.querySelector(".post");
const bootLog = []; // { label, status, raw, halt, progress, el } - status null = pending
const lineText = (e) => {
  if (e.raw) return e.label;
  // Pending line: bare label, plus a live counter when the stage reports
  // progress (e.g. "LOADING POLICIES [3/5]").
  if (e.status === null) return e.progress ? `${e.label} [${e.progress}]` : e.label;
  return `${e.label} `.padEnd(26, ".") + ` ${e.status}`;
};
const bootLine = (label) => {
  const entry = { label, status: null, raw: false, el: null, progress: null };
  bootLog.push(entry);
  const done = (status = "OK") => {
    entry.status = status;
    if (entry.el) entry.el.textContent = lineText(entry);
  };
  // Live sub-progress while the stage is still pending (honest path).
  done.progress = (p) => {
    entry.progress = p;
    if (entry.el && entry.status === null) entry.el.textContent = lineText(entry);
  };
  return done;
};
// Plain line with no dotted leader (header / error text).
const bootNote = (label) => {
  bootLog.push({ label, status: "", raw: true, el: null });
};
bootNote("Microduck BIOS v1.0");
bootLine("MEMORY CHECK")("640K OK");
bootLine("DUCK FIRMWARE")("PRESENT");

let bootDone = false;
let bootFailed = false;
let biosStarted = false;
const biosSleep = (ms) => new Promise((r) => setTimeout(r, ms));
// Fatal boot failure: freeze the sequence on the console. The BIOS never
// prints READY, never fades out and never cues the entrance - the halt
// screen IS the diagnostic surface. The welcome modal (if still up) is
// dropped so the console isn't stuck behind its blur.
function bootHalt(detail) {
  if (bootFailed || bootDone) return;
  bootFailed = true;
  bootNote(`>> ${detail}`);
  bootLog.push({ label: "SYSTEM HALTED", status: "", raw: true, el: null, halt: true });
  const modal = document.getElementById("welcome");
  if (modal) modal.hidden = true;
  playBios();
}
// Seeded LCG: the replay rhythm is random-feeling but identical on every
// load - real POST screens burst through most checks and stall on a few.
// Seed picked so the draw reads burst/burst/burst/stall/burst/stall.
let biosSeed = 11;
const biosRand = () => (biosSeed = (biosSeed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
async function playBios() {
  if (biosStarted) return;
  biosStarted = true;
  loadingEl.style.display = "flex";
  let i = 0;
  for (;;) {
    if (i < bootLog.length) {
      const entry = bootLog[i++];
      entry.el = document.createElement("div");
      if (entry.halt) entry.el.className = "halt";
      postEl.appendChild(entry.el);
      if (entry.status === null && !bootDone && !bootFailed) {
        // Honest mode: show the stage label and hold while it's really in
        // flight (its completer fills the status and .progress() updates
        // the live counter), cursor blinking via CSS.
        entry.el.textContent = lineText(entry);
        while (entry.status === null && !bootDone && !bootFailed) {
          await biosSleep(60);
          entry.el.textContent = lineText(entry); // live [n/m] counter
        }
        await biosSleep(90);
        entry.el.textContent = lineText(entry);
      } else if (entry.raw || entry.status === null) {
        // Header / note lines: quick, no dotted leader to animate.
        entry.el.textContent = lineText(entry);
        await biosSleep(20 + 60 * biosRand());
      } else {
        // Finished check: bursty POST pacing. Most lines snap in nearly
        // instantly; the occasional one stalls on a "slow check" - and on
        // a stall the dotted leader types out one dot at a time before
        // the status lands.
        const r = biosRand();
        const stall = r < 0.75 ? 20 * biosRand() : 150 + 250 * biosRand();
        if (stall < 45) {
          entry.el.textContent = lineText(entry);
          await biosSleep(stall);
        } else {
          const prefix = `${entry.label} `;
          const nDots = Math.max(26 - prefix.length, 3);
          entry.el.textContent = prefix;
          const per = stall / nDots;
          for (let d = 0; d < nDots; d++) {
            await biosSleep(per);
            entry.el.textContent += ".";
          }
          await biosSleep(40);
          entry.el.textContent = lineText(entry);
        }
      }
    } else if (bootDone) {
      break;
    } else {
      // Waiting for the next real stage. On a failed boot this parks the
      // console forever: everything queued (FAIL line, error detail,
      // SYSTEM HALTED) has been printed and nothing more will come.
      await biosSleep(bootFailed ? 500 : 60);
    }
  }
  const ready = document.createElement("div");
  ready.textContent = "READY.";
  postEl.appendChild(ready);
  await biosSleep(500);
  loadingEl.classList.add("off");
  // Wait out the overlay's 0.45 s opacity transition BEFORE cueing the
  // draw-in: starting it under the fading overlay hides the first (and
  // busiest) part of the animation and only the tail end shows.
  await biosSleep(500);
  loadingEl.style.display = "none";
  // Cue the world draw-in + duck scan-up on a fully black screen.
  // (Only ever reached after bootDone, so the ceremony module exists.)
  ceremony.startEntrance();
}
// Only the initial entry triggers the replay; reopening the modal from
// the brand later never replays (biosStarted latches).
if (window.__microduckEntered) playBios();
else document.addEventListener("microduck:enter", () => playBios(), { once: true });

// Surface boot failures on the page itself: a rejected top-level await
// otherwise leaves a dead page with no visible error. Gated on the boot
// still being in flight: post-boot async noise (ghost relay hiccups,
// audio autoplay rejections...) must NOT cue the BIOS replay early
// behind the welcome modal - that replay belongs to the "Waddle in"
// click alone. Boot-time errors route into the SYSTEM HALTED screen.
window.addEventListener("unhandledrejection", (e) => {
  console.error("[rl] unhandled rejection", e.reason);
  bootHalt(e.reason?.message || String(e.reason));
});
window.addEventListener("error", (e) => {
  console.error("[rl] window error", e.message);
  bootHalt(e.message);
});
// Halting at the failure site rather than relying on unhandledrejection:
// a rejected top-level await only surfaces through the dynamic import()
// chain in index.html, which does NOT reliably fire the window handler.
const traced = (label, p) => {
  const done = bootLine(label);
  return p.then(
    (v) => { done("OK"); return v; },
    (err) => {
      done("FAILED");
      console.error(`[rl] ${label} FAILED`, err);
      bootHalt(err?.message || String(err));
      throw err;
    },
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
// Works for both variants: the roller XML only differs by the ankle/wheel
// subtree (4 extra passive hinges), which the keyframe builder below
// handles by walking the joints in document order.
async function buildPhysicsXml(xmlFile) {
  const src = await (await fetch(signed(`${MODEL_DIR}/${xmlFile}`))).text();
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
  // STAND keyframe from mjlab's scene_walk.xml (STAND2 pose; the roller
  // scene_rollers.xml STAND uses the same trunk height and 14-joint pose).
  // qpos must cover every joint in document order: the 14 actuated hinges
  // take DEFAULT_POSE by name, anything else (the roller variant's passive
  // wheels) starts at zero. The ball's 7 free-joint values MUST be
  // appended or nq won't match and the model won't compile; parked 50 m
  // away = effectively absent.
  const qposFree = `${SPAWN_X} ${SPAWN_Y} 0.12 1 0 0 0`;
  const poseByName = new Map(JOINT_NAMES.map((n, i) => [n, DEFAULT_POSE[i]]));
  const qposJoints = [...doc.querySelectorAll("body > joint")]
    .map((j) => poseByName.get(j.getAttribute("name")) ?? 0)
    .join(" ");
  const pose14 = Array.from(DEFAULT_POSE).join(" ");
  const kf = doc.createElement("keyframe");
  kf.appendChild(el("key", {
    name: "STAND",
    qpos: `${qposFree} ${qposJoints} ${BALL_PARK_POS} 1 0 0 0`,
    ctrl: pose14,
  }));
  root.appendChild(kf);
  const meshFiles = [...doc.querySelectorAll("asset > mesh")].map((m) => m.getAttribute("file"));
  return { xml: new XMLSerializer().serializeToString(doc), meshFiles };
}

// ── Boot physics + policy in parallel with the render rig ──────────────
const [mujoco, { xml, meshFiles }, k] = await Promise.all([
  traced("MUJOCO WASM", loadMujoco()),
  traced("PHYSICS MJCF", buildPhysicsXml("robot_allcollisions.xml")),
  traced("KINEMATICS", loadKinematics(`${MODEL_DIR}/kinematics.json`)),
]);

const doneMeshes = bootLine("MESH ASSETS");
const vfs = new mujoco.MjVFS();
// One shared VFS for both variants; already-loaded files are skipped so
// the roller lazy-load only fetches its 5 new meshes.
const vfsFiles = new Set();
async function addMeshesToVfs(files) {
  await Promise.all(
    files.map(async (f) => {
      if (vfsFiles.has(f)) return;
      vfsFiles.add(f);
      // Same cache-busted URL as duck.js so the browser reuses the render
      // meshes instead of downloading the collision subset a second time.
      const buf = await (await fetch(signed(`${MODEL_DIR}/meshes/${f}?v=${MESH_VERSION}`), { cache: "force-cache" })).arrayBuffer();
      // meshdir="assets" in the MJCF, so the compiler looks up "assets/<f>".
      vfs.addBuffer(`assets/${f}`, new Uint8Array(buf));
    }),
  );
}
try {
  await addMeshesToVfs(meshFiles);
} catch (err) {
  doneMeshes("FAILED");
  bootHalt(err?.message || String(err));
  throw err;
}
doneMeshes(`${meshFiles.length} FILES`);

const sessions = {};
// Always boot on the classic (orange) colourway; the swatches re-skin live.
let currentVariant = DEFAULT_VARIANT;
const rigPromise = (async () => {
  const doneRig = bootLine("RENDER RIG");
  try {
    const rig = await buildRig(k, { materialForMesh: materialHookFor(VARIANTS[currentVariant]) });
    doneRig("OK");
    return rig;
  } catch (err) {
    doneRig("FAILED");
    bootHalt(err?.message || String(err));
    throw err;
  }
})();
// Boot policies with a live [n/5] counter on the BIOS line; any single
// session failure marks the whole line FAILED (the halt detail line names
// the actual error).
const donePolicies = bootLine("LOADING POLICIES");
const sessionOpts = { executionProviders: ["wasm"] };
let policiesLoaded = 0;
const bootPolicy = (url) =>
  ort.InferenceSession.create(signed(url), sessionOpts).then((s) => {
    donePolicies.progress(`${++policiesLoaded}/5`);
    return s;
  });
try {
  [sessions.walk, sessions.sitstand, sessions.roll, sessions.kickL, sessions.kickR] =
    await Promise.all([
      bootPolicy(POLICIES.walk),
      bootPolicy(POLICIES.sitstand),
      bootPolicy(POLICIES.roll),
      bootPolicy(POLICIES.kickL),
      bootPolicy(POLICIES.kickR),
    ]);
} catch (err) {
  donePolicies("FAILED");
  bootHalt(err?.message || String(err));
  throw err;
}
donePolicies("5/5");

const doneCompile = bootLine("COMPILING PHYSICS");
let model, data;
try {
  model = mujoco.MjModel.from_xml_string(xml, vfs);
  data = new mujoco.MjData(model);
} catch (err) {
  doneCompile("FAILED");
  bootHalt(err?.message || String(err));
  throw err;
}
doneCompile("COMPILED");

// Addresses resolved once per compiled variant. qpos/qvel/sensordata views
// are re-read at each use: the WASM heap can grow and detach earlier
// TypedArray views.
// NOTE: unlike the Python bindings, these accessor fields are plain numbers.
const JOINT_SET = new Set(JOINT_NAMES);
function resolveAddrs(model, kin) {
  return {
    qposAdr: JOINT_NAMES.map((n) => model.jnt(n).qposadr),
    dofAdr: JOINT_NAMES.map((n) => model.jnt(n).dofadr),
    gyroAdr: model.sensor("imu_ang_vel").adr,
    trunkId: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, "trunk_base"),
    standKeyId: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY.value, "STAND"),
    ballQposAdr: model.jnt("ball_freejoint").qposadr,
    ballDofAdr: model.jnt("ball_freejoint").dofadr,
    // Unactuated hinges (the roller variant's 4 passive wheels): not in
    // the obs or ctrl, but synced to the render rig so the wheels spin.
    extraJoints: kin.bodies
      .filter((b) => b.joint && b.joint.type === "hinge" && !JOINT_SET.has(b.joint.name))
      .map((b) => ({ name: b.joint.name, adr: model.jnt(b.joint.name).qposadr })),
  };
}
// Active-variant address block, swapped wholesale by activateLoco.
let { qposAdr, dofAdr, gyroAdr, trunkId, standKeyId, ballQposAdr, ballDofAdr, extraJoints } =
  resolveAddrs(model, k);

// Locomotion variants stay resident once built (model + data + rig +
// addresses); legs is registered when its render rig resolves below.
const locos = {};
let loco = "legs"; // "legs" | "rollers"
const velLims = () => (loco === "rollers"
  ? [RVEL_FWD, RVEL_BACK, RVEL_ANG]
  : [VEL_FWD, VEL_BACK, VEL_ANG]);

let uiReady = false;
const lastAction = new Float32Array(NUM_JOINTS);
const obs = new Float32Array(OBS_SIZE);
const cmd = new Float32Array(CMD_SIZE); // [vx, vy, wz, head(4), body(6)]
// Input controller (controls/controller.js): keyboard + gamepad sources
// merged into one continuous command + discrete action surface. Registered
// in priority order - live pad sticks win over held keys, exactly the old
// padActive arbitration. Constructed up here because the control loop reads
// it via effectiveCmd on the very first control step; the sources' event
// listeners are only armed by controller.init() in the input-wiring
// section below, where the raw listeners historically attached.
const kbSource = new KeyboardSource({ getVelocityLimits: () => velLims() });
const padSource = new GamepadSource({ getVelocityLimits: () => velLims() });
const touchSource = new TouchSource({ getVelocityLimits: () => velLims() });
// Keyboard last: it reads zero when idle, so it doubles as the fallback.
const controller = new Controller({ sources: [padSource, touchSource, kbSource] });
// Right-stick camera state, read by renderStats before the camera-orbit
// section below has evaluated.
let padOrbitLive = false; // HUD: RS keycap lit while deflected
// Robot input gate: twist commands, mode changes, rolls, kicks and ball
// spawns all stay inert until the entrance sequence has fully played out
// (ceremony.drive flips this exactly when the choreography completes).
// Re-engaged by every post-entrance reset for the duck's re-scan.
let inputLocked = true;
// Cutscene + ball-actor modules, assigned once the scene exists. Boot
// resetSim runs before that (physics only); later kills go through these.
let ceremony = null;
let ball = null;
// Comic sticker popups (stickers.js). Nullable on purpose: every hook is a
// one-line `stickers?.pop(...)`, so deleting the single import line below
// (search for initStickers) removes the feature without breaking anything.
let stickers = null;
// HUD: Space keycap flash. Written by the respawn ceremony; read by
// renderStats.
let resetFlashAt = -Infinity;

// "walk" is the main velocity-tracking mode in BOTH variants (legs walking
// policy or roller drive policy - activeSession() picks); "crouch" is the
// roller-only one-shot; the rest are legs-only.
let mode = "walk"; // "walk" | "sitstand" | "roll" | "kickL" | "kickR" | "crouch"
let sitFlag = 0;
const isKick = () => mode === "kickL" || mode === "kickR";
// Local-only kickable ball: false while parked at the keyframe spot
// (mesh hidden), true once popped in front of the duck.
let ballActive = false;

// The twist the policy actually receives: the controller's merged command
// (live gamepad sticks win over held keys via source arbitration). No
// input means zero command - the duck stands in place.
// Shared by buildObs and the HUD mini-sticks so they can never disagree.
// Mid-roll every movement input is ignored (zero twist) until the roll
// hands back to walk on its own - steering would only knock the roll over.
const ZERO_CMD = new Float32Array(3);
function effectiveCmd() {
  if (inputLocked || mode === "roll" || mode === "crouch" || isKick() || postKickLock > 0)
    return ZERO_CMD;
  return controller.getCommand();
}
// One-shot roll tracking: trigger time + whether the trunk actually
// tipped over yet. Set by triggerRoll, cleared when we hand back to walk.
let rollRun = null;
let rollSource = "kb"; // which hint lights up: keyboard Space or pad X
// One-shot crouch-glide tracking (roller variant): phase 0 -> 0.7 over
// 3.5 s, driven per control step. Shares rollSource for keycap lighting
// (same R / pad-X trigger).
let crouchRun = null;
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
let fallenSince = null; // wall-clock start of the current fallen spell
function clearModeTimers() {
  clearTimeout(sitTimer); sitTimer = null;
  clearTimeout(standTimer); standTimer = null;
}

function resetSim() {
  // Single reset path: Space, fall-kill, failed roll, loco switch. A
  // reset is a full transition to the walking stand: cancel any roll in
  // flight and any pending sit/stand hand-over, or they'd replay on the
  // freshly reset duck.
  clearModeTimers();
  rollRun = null;
  kickRun = null;
  crouchRun = null;
  postKickLock = 0;
  fallenSince = null;
  mode = "walk";
  mujoco.mj_resetDataKeyframe(model, data, standKeyId);
  mujoco.mj_forward(model, data);
  lastAction.fill(0);
  sitFlag = 0;
  // Park the ball in physics immediately; if it was on screen, the
  // reverse scan peels it away at its last pose. A queued B-respawn is
  // cancelled: a reset means no ball.
  ball?.despawn({ cancelQueued: true, parkPhysics: parkBallPhysics });
  ballActive = false;
  // Buttons reflect sitFlag; keep them honest after auto-resets.
  if (uiReady) syncButtons();
  ceremony?.playRespawn();
}
resetSim();

// Pop / respawn the ball ~0.35 m in front of the duck, with a small
// random heading + distance jitter so repeated pops land somewhere
// nearby instead of always on the same spot. qvel is zeroed so a respawn
// doesn't carry the old momentum.
// If the ball is already on screen, peel it away first (reverse scan)
// and pop the new one when that finishes - same appear/disappear pair
// as the duck's wireframe ceremony.
function parkBallPhysics() {
  const qpos = data.qpos, qvel = data.qvel;
  qpos[ballQposAdr] = 50;
  qpos[ballQposAdr + 1] = 0;
  qpos[ballQposAdr + 2] = BALL_RADIUS;
  qpos[ballQposAdr + 3] = 1;
  qpos[ballQposAdr + 4] = 0;
  qpos[ballQposAdr + 5] = 0;
  qpos[ballQposAdr + 6] = 0;
  for (let i = 0; i < 6; i++) qvel[ballDofAdr + i] = 0;
  mujoco.mj_forward(model, data);
  ballActive = false;
}

function spawnBall(opts = {}) {
  if (inputLocked && !opts.fromQueue) return;
  if (!ball) return;
  if (ball.visual !== "hidden") {
    ball.queueRespawn();
    ball.despawn({ parkPhysics: parkBallPhysics });
    return;
  }
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
  // Snap the mesh to the new pose BEFORE the scan starts: the FX
  // recomputes its bbox from the live mesh.
  ball.poseFromQpos(qpos, ballQposAdr);
  ball.appear();
  stickers?.pop("spawn");
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
  // command: walking/drive use the twist; sitstand uses cmd[0] as the
  // posture flag; the crouch-glide one-shot carries its phase encoding in
  // the vel slots (ground-pick convention: [cos, sin, 0]).
  cmd.fill(0, 0, 3);
  if (mode === "sitstand") {
    cmd[0] = sitFlag;
  } else if (mode === "crouch" && crouchRun) {
    const a = 2 * Math.PI * crouchRun.phase;
    cmd[0] = Math.cos(a);
    cmd[1] = Math.sin(a);
  } else {
    // walk uses the live twist; roll and kick see all-zero commands
    // (via effectiveCmd), matching how they were trained.
    const c = effectiveCmd();
    cmd[0] = c[0]; cmd[1] = c[1]; cmd[2] = c[2];
  }
  for (let c = 0; c < CMD_SIZE; c++) obs[i++] = cmd[c];
  return obs;
}

// The ONNX session for the current mode: in the roller variant the main
// velocity mode runs the drive (skating) policy instead of the walker.
const activeSession = () =>
  sessions[loco === "rollers" && mode === "walk" ? "drive" : mode];

// ── Control loop (50 Hz, async because ONNX inference is async) ────────
let ctrlHz = 0;

// Dead pose: walk/sitstand have no get-up skill, so a kill here is just
// a resetSim (same ceremony as Space). Height alone would false-positive
// on a deep sit, so "fallen" = trunk tilted past ~60 deg (projected
// gravity z above -0.5) or sunk below the floor. NaN/Inf is a solver
// explosion: no grace, reset on the spot. The roll tumbles the trunk on
// purpose and recovers on its own, so it gets a much longer grace window
// before we call it stuck.
function poseIsDead() {
  const z = data.qpos[2];
  const gz = obs[5]; // projected gravity z, from the last obs
  if (!Number.isFinite(z) || !Number.isFinite(gz)) return "exploded";
  if (gz > -0.5 || z < 0.02) return "fallen";
  return null;
}

async function controlStep() {
  const feeds = { obs: new ort.Tensor("float32", buildObs(), [1, OBS_SIZE]) };
  const out = await activeSession().run(feeds);
  const act = out.actions.data;
  lastAction.set(act);
  const ctrl = data.ctrl;
  for (let j = 0; j < NUM_JOINTS; j++) ctrl[j] = DEFAULT_POSE[j] + act[j] * ACTION_SCALE;
  for (let s = 0; s < DECIMATION; s++) mujoco.mj_step(model, data);

  const death = poseIsDead();
  if (death === "exploded") {
    resetSim();
  } else if (death === "fallen") {
    const now = performance.now();
    const graceMs = mode === "roll" ? 5000 : 1000;
    fallenSince ??= now;
    if (now - fallenSince > graceMs) resetSim();
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

  // Crouch-glide one-shot: advance the trained phase clock and hand back
  // to the drive policy at the runtime's cycle end (0.7 x 5 s = 3.5 s:
  // sink, glide low, stand back up). lastAction stays continuous, same as
  // every other policy swap.
  if (mode === "crouch" && crouchRun) {
    crouchRun.phase += CTRL_DT / CROUCH_PERIOD_S;
    if (crouchRun.phase >= CROUCH_END_PHASE) {
      crouchRun = null;
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
      // (it has no get-up skill). Same ceremony as Space / fall-kill.
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
// Boot framing translated onto the spawn cell (the orbit target follows
// below), so the follow-cam has nothing to drift toward during boot.
camera.position.set(SPAWN_X + 0.55, 0.35, -SPAWN_Y + 0.7);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
renderer.setClearColor(0x08080c, 1);
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
      uSection: { value: GRID_SECTION },
      uCellColor: { value: new THREE.Color(0x8e8371) },
      uSectionColor: { value: new THREE.Color(0xffb366) },
      uFadeDist: { value: 3.0 },
      uFocus: { value: new THREE.Vector3() },
      // Entrance draw-in progress; 1 = steady state (branch skipped).
      // Starts at 0: the world stays hidden behind the welcome modal and
      // the BIOS readout until playBios cues startEntrance.
      uReveal: { value: 0.0 },
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
      uniform float uCell, uSection, uFadeDist, uReveal;
      uniform vec3 uCellColor, uSectionColor, uFocus;
      // Tron-style line: a thicker antialiased core plus a faint, much
      // wider halo added on top (squared falloff keeps it a whisper of a
      // glow rather than a bloom wash).
      float lineProf(float g) {
        float core = 1.0 - smoothstep(0.0, 1.8, g);
        float halo = 1.0 - smoothstep(0.0, 7.0, g);
        return core + halo * halo * 0.22;
      }
      float gridLine(vec2 p, float size) {
        vec2 r = p / size;
        vec2 g = abs(fract(r - 0.5) - 0.5) / fwidth(r);
        return lineProf(min(g.x, g.y));
      }
      // Entrance draw-in: one family of parallel lines, drawn line by line.
      // id picks the line, "along" runs down its length. Each line waits
      // out its own hashed delay, then extends from the origin outward with
      // a hard front. Returns (mask, head): head marks the bright segment
      // right behind the draw front while the line is still growing.
      vec2 drawLine(float id, float along, float t0, float spread, float dur, float maxLen) {
        float jit = fract(sin(id * 127.1) * 43758.5453);
        float grow = clamp((uReveal - t0 - jit * spread) / dur, 0.0, 1.0);
        float len = grow * maxLen;
        float a = abs(along);
        float mask = 1.0 - smoothstep(len - 0.05, len, a);
        float head = (1.0 - smoothstep(0.0, 0.6, len - a)) * mask
                   * step(0.001, grow) * (1.0 - step(0.999, grow));
        return vec2(mask, head);
      }
      void main() {
        float cell = gridLine(vWorld.xz, uCell);
        // Section lattice shifted half a cell: with 5 sections across the
        // 3 m arena (odd count) this centers a CELL on the origin and puts
        // section lines exactly on the walls at +-1.5.
        vec2 pSec = vWorld.xz + 0.5 * uSection;
        float section = gridLine(pSec, uSection);
        float d = distance(vWorld.xz, uFocus.xz);
        float fade = pow(clamp(1.0 - d / uFadeDist, 0.0, 1.0), 1.6);
        vec3 col = mix(uCellColor, uSectionColor, clamp(section, 0.0, 1.0));
        float alpha = min(max(section * 0.6, cell * 0.4) * fade, 1.0);
        // Entrance: only the bright section lines get the line-by-line draw
        // (staggered, with a hot draw head); the fine cells just fade in
        // over the reveal's second half - drawing every small line reads as
        // visual noise. lineProf(min(gx, gy)) == max of per-axis profiles
        // (profile is monotonic) and the cell fade lands on exactly the
        // steady-state cell term, so at uReveal 1 this branch equals the
        // formula above exactly (and is skipped).
        if (uReveal < 1.0) {
          vec2 rs = pSec / uSection;
          vec2 gs = abs(fract(rs - 0.5) - 0.5) / fwidth(rs);
          // Const-x lines run along z and vice versa; the offset
          // decorrelates the two families' hashed delays.
          vec2 sx = drawLine(floor(rs.x + 0.5), vWorld.z, 0.00, 0.30, 0.35, 8.0);
          vec2 sz = drawLine(floor(rs.y + 0.5) + 57.0, vWorld.x, 0.05, 0.30, 0.35, 8.0);
          float secR = max(lineProf(gs.x) * sx.x, lineProf(gs.y) * sz.x);
          float cellR = cell * smoothstep(0.5, 1.0, uReveal);
          float headGlow = max(lineProf(gs.x) * sx.y, lineProf(gs.y) * sz.y);
          col = mix(uCellColor, uSectionColor, clamp(secR, 0.0, 1.0));
          alpha = min(max(secR * 0.6, cellR * 0.4) * fade, 1.0);
          // Bright draw head: a short white-hot tip sells the "drawing" read.
          headGlow = clamp(headGlow, 0.0, 1.0);
          col = mix(col, vec3(1.0, 0.86, 0.55), headGlow * 0.8);
          alpha = min(alpha + headGlow * fade * 0.5, 1.0);
        }
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
      uSection: { value: GRID_SECTION },
      uCellColor: { value: new THREE.Color(0x8e8371) },
      uSectionColor: { value: new THREE.Color(0xffb366) },
      // Gentler radial fade than the floor: the walls sit 1.5+ m from the
      // duck by construction and would vanish with the floor's 3 m fade.
      uFadeDist: { value: 5.0 },
      uFocus: { value: new THREE.Vector3() },
      uWallH: { value: ARENA_WALL_H },
      uAlongX: { value: alongX ? 1.0 : 0.0 },
      // Entrance draw-in progress; 1 = steady state (branch skipped).
      // Starts at 0, same as the floor grid: hidden until startEntrance.
      uReveal: { value: 0.0 },
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
      uniform float uCell, uSection, uFadeDist, uWallH, uAlongX, uReveal;
      uniform vec3 uCellColor, uSectionColor, uFocus;
      // Same Tron-style core + faint halo as the floor grid.
      float lineProf(float g) {
        float core = 1.0 - smoothstep(0.0, 1.8, g);
        float halo = 1.0 - smoothstep(0.0, 7.0, g);
        return core + halo * halo * 0.22;
      }
      float gridLine(vec2 p, float size) {
        vec2 r = p / size;
        vec2 g = abs(fract(r - 0.5) - 0.5) / fwidth(r);
        return lineProf(min(g.x, g.y));
      }
      // Same line-by-line draw as the floor grid (see its comments).
      vec2 drawLine(float id, float along, float t0, float spread, float dur, float maxLen) {
        float jit = fract(sin(id * 127.1) * 43758.5453);
        float grow = clamp((uReveal - t0 - jit * spread) / dur, 0.0, 1.0);
        float len = grow * maxLen;
        float a = abs(along);
        float mask = 1.0 - smoothstep(len - 0.05, len, a);
        float head = (1.0 - smoothstep(0.0, 0.35, len - a)) * mask
                   * step(0.001, grow) * (1.0 - step(0.999, grow));
        return vec2(mask, head);
      }
      void main() {
        // Wall surface coords: the in-plane horizontal world axis + height.
        float h = mix(vWorld.z, vWorld.x, uAlongX);
        vec2 p = vec2(h, vWorld.y);
        float cell = gridLine(p, uCell);
        // Horizontal axis shifted half a section to match the floor's odd
        // lattice (vertical section lines meet the floor's at the base);
        // the height axis keeps its base line at y = 0.
        vec2 pSec = vec2(p.x + 0.5 * uSection, p.y);
        float section = gridLine(pSec, uSection);
        float d = distance(vWorld.xz, uFocus.xz);
        float fade = pow(clamp(1.0 - d / uFadeDist, 0.0, 1.0), 1.6);
        float vert = 1.0 - clamp(vWorld.y / uWallH, 0.0, 1.0);
        vec3 col = mix(uCellColor, uSectionColor, clamp(section, 0.0, 1.0));
        float alpha = min(max(section * 0.9, cell * 0.6) * fade * (0.3 + 0.7 * vert), 1.0);
        // Entrance: section lines only - horizontals zip out from the
        // wall's center, verticals rise from the ground, each with a
        // hashed delay; the fine cells fade in over the reveal's second
        // half. Same steady-state equivalence argument as the floor grid.
        if (uReveal < 1.0) {
          vec2 rs = pSec / uSection;
          vec2 gs = abs(fract(rs - 0.5) - 0.5) / fwidth(rs);
          // Const-height lines run along h (grow from center outward);
          // const-h lines run along y (grow up from the ground).
          vec2 sh = drawLine(floor(rs.y + 0.5), p.x, 0.00, 0.30, 0.40, 2.0);
          vec2 sv = drawLine(floor(rs.x + 0.5) + 31.0, p.y, 0.30, 0.30, 0.30, uWallH);
          float secR = max(lineProf(gs.y) * sh.x, lineProf(gs.x) * sv.x);
          float cellR = cell * smoothstep(0.5, 1.0, uReveal);
          float headGlow = max(lineProf(gs.y) * sh.y, lineProf(gs.x) * sv.y);
          col = mix(uCellColor, uSectionColor, clamp(secR, 0.0, 1.0));
          alpha = min(max(secR * 0.9, cellR * 0.6) * fade * (0.3 + 0.7 * vert), 1.0);
          headGlow = clamp(headGlow, 0.0, 1.0);
          col = mix(col, vec3(1.0, 0.86, 0.55), headGlow * 0.8);
          alpha = min(alpha + headGlow * fade * 0.5, 1.0);
        }
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

let rig = await rigPromise;
scene.add(rig.placer);
let trunkGroup = rig.bodies.get("trunk_base");
locos.legs = {
  model, data, rig, trunkGroup,
  qposAdr, dofAdr, gyroAdr, trunkId, standKeyId, ballQposAdr, ballDofAdr, extraJoints,
};

// ── Locomotion variant switching (legs <-> rollers) ─────────────────────
// The roller stack (XML + 5 extra meshes + kinematics + 2 ONNX policies)
// is lazy-loaded on the first switch so the default boot stays untouched,
// then kept resident: switching back and forth only swaps references.
let rollersLoading = null;
function ensureRollers() {
  rollersLoading ??= (async () => {
    const [{ xml: rXml, meshFiles: rMeshFiles }, rk] = await Promise.all([
      buildPhysicsXml("robot_allcollisions_rollers.xml"),
      loadKinematics(`${MODEL_DIR}/kinematics_rollers.json`),
    ]);
    const [rRig, sDrive, sCrouch] = await Promise.all([
      buildRig(rk, { materialForMesh: materialHookFor(VARIANTS[currentVariant]) }),
      ort.InferenceSession.create(signed(POLICIES.drive), sessionOpts),
      ort.InferenceSession.create(signed(POLICIES.crouch), sessionOpts),
      addMeshesToVfs(rMeshFiles),
    ]);
    sessions.drive = sDrive;
    sessions.crouch = sCrouch;
    const rModel = mujoco.MjModel.from_xml_string(rXml, vfs);
    const rData = new mujoco.MjData(rModel);
    locos.rollers = {
      model: rModel, data: rData, rig: rRig, trunkGroup: rRig.bodies.get("trunk_base"),
      ...resolveAddrs(rModel, rk),
    };
  })();
  return rollersLoading;
}

function activateLoco(name) {
  const L = locos[name];
  loco = name;
  scene.remove(rig.placer);
  ({ model, data, rig, trunkGroup, qposAdr, dofAdr, gyroAdr, trunkId,
     standKeyId, ballQposAdr, ballDofAdr, extraJoints } = L);
  // The rig may have been built (or last shown) under another colourway.
  applyVariant(rig, currentVariant);
  scene.add(rig.placer);
  document.body.classList.toggle("rollers", name === "rollers");
  resetSim();
  if (uiReady) syncLocoHints();
}

// OSD line while the roller stack streams in, BIOS-style.
let osdLoadEl = null;
let locoSwitching = false;
async function setLoco(name, { force = false } = {}) {
  if (name !== "legs" && name !== "rollers") return;
  if (loco === name || locoSwitching) return;
  if (!force && (inputLocked || rollRun || kickRun || crouchRun || standTimer)) return;
  locoSwitching = true;
  try {
    if (name === "rollers" && !locos.rollers) {
      osdLoadEl?.removeAttribute("hidden");
      await ensureRollers();
    }
    activateLoco(name);
  } catch (e) {
    rollersLoading = null;
    console.error("[rl] roller switch failed", e);
  } finally {
    osdLoadEl?.setAttribute("hidden", "");
    locoSwitching = false;
  }
}

async function toggleLoco() {
  await setLoco(loco === "legs" ? "rollers" : "legs");
}

// ── Cutscenes (entrance + respawn) ──────────────────────────────────────
// Owned by ceremony.js. Physics reset stays in resetSim; this is the
// camera + materialization layer. Boot-hidden: bind parks the duck clip
// below the feet so nothing shows through the welcome modal / BIOS.
const fx = await import(signed(`./fx/fx-wireframe.js?v=${SELF_V}`));
const { createCeremony, CAM_RESET_S } = await import(signed(`./ceremony.js?v=${SELF_V}`));
ceremony = createCeremony({
  THREE, scene, camera, renderer, fx,
  getRig: () => rig,
  grid, wallMats,
  syncRig, startCameraReset,
  setLocked: (v) => {
    inputLocked = v;
    controller.setLocked(v);
    // A ball is always in play: pop one the moment the entrance or a
    // respawn ceremony hands control back (resets park the previous
    // ball, so this re-pops it fresh in front of the duck).
    if (!v && ball && !ballActive) spawnBall({ fromQueue: true });
  },
  flashReset: () => { resetFlashAt = performance.now(); },
});
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
ballMesh.userData.meshName = "ball";
// The 48x32 render sphere is far too dense for the wireframe scan (it
// reads as a solid glowing blob); the FX overlay uses this geodesic
// stand-in instead - 80 triangles, clean hologram lines.
ballMesh.userData.fxWireGeometry = new THREE.IcosahedronGeometry(BALL_RADIUS, 1);
ballMesh.visible = false;
ballGroup.add(ballMesh);
scene.add(ballGroup);

const { createBallActor } = await import(signed(`./ball-actor.js?v=${SELF_V}`));
ball = createBallActor({
  THREE, scene, camera, renderer, fxModule: fx, mesh: ballMesh, group: ballGroup,
});

// Comic sticker popups on game events (kick / quack / roll / ball spawn /
// ghost join). Fully self-contained DOM overlay: delete this import to
// remove the feature (the `stickers?.pop` hooks then no-op).
// Stickers disabled for now - uncomment to re-enable.
// stickers = (await import(signed(`./stickers.js?v=${SELF_V}`)))
//   .initStickers({ signed, isLocked: () => inputLocked });

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(SPAWN_X, 0, -SPAWN_Y); // orbit around the spawn cell
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
// Heading hysteresis (Schmitt trigger): the walking gait wiggles the trunk
// yaw substantially every step (measured ~±14 deg at full speed) and
// tracking it 1:1 makes the camera sway left-right constantly. Two layers:
//   1. chaseYawSmooth: slow EMA of the trunk yaw - the gait's oscillation
//      is symmetric so this is the duck's MEAN heading, ~steady while
//      walking straight, moving cleanly during a real sustained turn.
//   2. chaseYawFollow: what the camera frames. Frozen until the smooth
//      heading deviates beyond ENGAGE, then eases toward it and freezes
//      again below RELEASE (classic Schmitt).
// An intentional turn (non-zero wz command from keys or pad) bypasses the
// deadband immediately and both layers track fast, so full-speed turns
// stay responsive.
let chaseYawSmooth = 0; // EMA of trunk yaw (spawn yaw = 0)
let chaseYawFollow = 0; // heading the camera actually frames
let chaseYawTracking = false;
const CHASE_YAW_SMOOTH_EASE = 0.04; // per-frame EMA, ~0.4 s time constant
const CHASE_YAW_ENGAGE = 0.17; // rad, ~10 deg: start re-tracking
const CHASE_YAW_RELEASE = 0.03; // rad, ~1.7 deg: stop once realigned
const CHASE_YAW_EASE = 0.10; // per-frame ease during hysteresis catch-up
// During a commanded turn the heading chain must not stack lag on top of
// the camera position lerp (rollers turn at ~70 deg/s): near-snap ease,
// the position lerp alone provides the smoothing, like pre-hysteresis.
const CHASE_YAW_EASE_TURN = 0.5;
const wrapPi = (a) => Math.atan2(Math.sin(a), Math.cos(a));
function updateChaseCam() {
  // Reset glide: one clean tween from wherever the camera is back to the
  // home framing (see startCameraReset). Runs instead of the chase logic
  // and hands control back to it on landing - the destination IS the
  // chase cam's ideal point, so the handoff is seamless.
  if (camResetT0 !== null) {
    // Any detach (drag, right stick, C/R3 toggle) cancels the glide and
    // gives the camera straight back to the user.
    if (!chaseCam) { camResetT0 = null; return; }
    const t = (performance.now() - camResetT0) / 1000 / CAM_RESET_S;
    const e = t >= 1 ? 1 : t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    camera.position.lerpVectors(_camFrom, _camTo, e);
    controls.target.lerpVectors(_tgtFrom, _tgtTo, e);
    camera.lookAt(controls.target);
    if (t >= 1) camResetT0 = null;
    return;
  }
  if (!chaseCam) return;
  const qpos = data.qpos;
  let rawYaw;
  if (mode === "roll" || isKick()) {
    rawYaw = chaseHeldYaw;
  } else {
    rawYaw = Math.atan2(
      2 * (qpos[3] * qpos[6] + qpos[4] * qpos[5]),
      1 - 2 * (qpos[5] * qpos[5] + qpos[6] * qpos[6]),
    );
    chaseHeldYaw = rawYaw;
  }
  // Schmitt heading follow (see constants above). "turning" reads the raw
  // per-source wz commands (not the locked/merged view) so an intentional
  // turn engages on the first frame.
  const turning = controller.sources.some((s) => Math.abs(s.command[2]) > 0.05);
  chaseYawSmooth = wrapPi(
    chaseYawSmooth +
      wrapPi(rawYaw - chaseYawSmooth) * (turning ? CHASE_YAW_EASE_TURN : CHASE_YAW_SMOOTH_EASE),
  );
  const yawErr = wrapPi(chaseYawSmooth - chaseYawFollow);
  if (turning || Math.abs(yawErr) > CHASE_YAW_ENGAGE) chaseYawTracking = true;
  if (chaseYawTracking) {
    chaseYawFollow = wrapPi(
      chaseYawFollow + yawErr * (turning ? CHASE_YAW_EASE_TURN : CHASE_YAW_EASE),
    );
    if (!turning && Math.abs(yawErr) < CHASE_YAW_RELEASE) chaseYawTracking = false;
  }
  const yaw = chaseYawFollow;
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

// ── Camera reset (owned by the respawn ceremony) ────────────────────────
// Glides back to the page-load framing: the chase cam's ideal point
// behind the duck's spawn heading, at the boot orbit distance. The
// distance is captured here, before any user input can zoom; the rest
// of the destination is computed live at reset time from the freshly
// reset qpos, so it lands correctly in both legs and roller modes.
const CAM_HOME_DIST = camera.position.distanceTo(controls.target);
let camResetT0 = null; // wall-clock start while the glide is playing
const _camFrom = new THREE.Vector3(), _camTo = new THREE.Vector3();
const _tgtFrom = new THREE.Vector3(), _tgtTo = new THREE.Vector3();
function startCameraReset() {
  const qpos = data.qpos;
  const yaw = Math.atan2(
    2 * (qpos[3] * qpos[6] + qpos[4] * qpos[5]),
    1 - 2 * (qpos[5] * qpos[5] + qpos[6] * qpos[6]),
  );
  chaseHeldYaw = yaw; // keep the post-glide chase heading coherent
  chaseYawSmooth = yaw; // hysteresis state lands with the glide
  chaseYawFollow = yaw;
  chaseYawTracking = false;
  _tgtTo.set(qpos[0], qpos[2], -qpos[1]); // trunk at spawn, MJCF -> three
  const horiz = CAM_HOME_DIST * Math.cos(CHASE_PITCH);
  const vert = CAM_HOME_DIST * Math.sin(CHASE_PITCH);
  _camTo.set(
    _tgtTo.x - Math.cos(yaw) * horiz,
    _tgtTo.y + vert,
    _tgtTo.z + Math.sin(yaw) * horiz,
  );
  _camFrom.copy(camera.position);
  _tgtFrom.copy(controls.target);
  camResetT0 = performance.now();
  chaseCam = true; // reset always re-attaches the chase cam
}

// Pause: while the menu is up over a live game, keys belong to the menu.
const setInputLock = (v) => { inputLocked = v; controller.setLocked(v); };
document.addEventListener("microduck:menu-open", () => {
  if (ceremony.entranceDone) setInputLock(true);
});
document.addEventListener("microduck:menu-close", () => {
  if (ceremony.entranceDone && !ceremony.respawnActive) setInputLock(false);
});

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
  // Passive hinges (roller wheels): purely visual, driven straight from qpos.
  for (const ej of extraJoints) setJoint(rig, ej.name, qpos[ej.adr]);
  // Ball: live follows qpos; ghost freeze is owned by the ball actor
  // (physics already parked, mesh holds last pose during reverse scan).
  if (ball) ball.sync(qpos, ballQposAdr, ballActive);
  // Follow cam: ease the orbit target toward the trunk and translate the
  // camera by the same delta, so the camera-to-duck distance and viewing
  // angle stay constant while the duck walks. Mouse orbit/zoom still work:
  // they only change the (preserved) camera-target offset. Paused while
  // the reset glide owns the camera (it tweens the target itself).
  if (camResetT0 === null) {
    _target.set(qpos[0], qpos[2], -qpos[1]);
    _follow.copy(_target).sub(controls.target);
    // Horizontal follow at the usual rate; vertical much slower so the
    // per-step gait bob doesn't nod the frame (sit/crouch height changes
    // still settle in, just over ~1 s instead of instantly).
    _follow.x *= 0.06;
    _follow.z *= 0.06;
    _follow.y *= 0.015;
    controls.target.add(_follow);
    camera.position.add(_follow);
  }
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
  stickers?.pop("quack");
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
    loco: el("key-loco"),
    padX: el("key-pad-x"), padSit: el("key-pad-sit"),
    padRun: el("key-pad-run"), padRt: el("key-pad-rt"),
    padRb: el("key-pad-rb"), padLb: el("key-pad-lb"),
    padY: el("key-pad-y"), padRs: el("key-pad-rs"),
    padR3: el("key-pad-r3"),
  };
osdLoadEl = el("osd-load");
// Roller mode re-labels the trick hints (R / pad X trigger the crouch-glide
// instead of the roll) and greys the legs-only actions via body.rollers.
const rollLabelEl = el("key-roll-label");
const padXLabelEl = el("key-pad-x-label");
function syncLocoHints() {
  const rollers = loco === "rollers";
  if (rollLabelEl) rollLabelEl.textContent = rollers ? "crouch" : "roll";
  if (padXLabelEl) padXLabelEl.textContent = rollers ? "crouch" : "roll";
}
const STICK_R = 15; // px, max dot travel inside the 46px stick circle
let ballFlashAt = -Infinity;
let padYFlashAt = -Infinity;

// Bottom-right OSD extras: render rate (EMA over frame deltas), live ground
// speed off the freejoint, and an odometer integrating horizontal trunk
// travel. Teleport-sized jumps (resets, loco swaps) don't count as travel.
let fpsEma = 60;
let fpsLastT = performance.now();
let odoM = 0;
let odoX = null, odoY = null;

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

  const now = performance.now();
  const dtF = (now - fpsLastT) / 1000;
  fpsLastT = now;
  if (dtF > 0 && dtF < 0.5) fpsEma += (1 / dtF - fpsEma) * 0.05;
  const stepD = (odoX === null) ? 0 : Math.hypot(data.qpos[0] - odoX, data.qpos[1] - odoY);
  if (stepD < 0.05) odoM += stepD; // plausible per-frame travel only
  odoX = data.qpos[0];
  odoY = data.qpos[1];
  const spd = Math.hypot(data.qvel[0], data.qvel[1]);

  // Bottom-right telemetry stack, quietest line last: peers (only when
  // someone else is around), then speed + odometer, then the loop rates.
  const odo = odoM < 1000 ? `${odoM.toFixed(1)}M` : `${(odoM / 1000).toFixed(2)}KM`;
  const lines = [];
  if (peers) lines.push(`${peers + 1} ONLINE`);
  lines.push(`${spd.toFixed(2)}M/S \u00b7 ODO ${odo}`);
  lines.push(`FPS ${Math.round(fpsEma)} \u00b7 CTRL ${ctrlHz.toFixed(0)}HZ`);
  ctrlHzEl.textContent = lines.join("\n");

  // Mini sticks: the dot mirrors the effective twist, lit yellow while the
  // user is actually driving. Normalized against the ACTIVE variant's
  // velocity limits so full deflection reads the same in both.
  const manual = controller.anyActive() && mode !== "roll";
  const [limF, limB, limA] = velLims();
  const yN = vx >= 0 ? vx / limF : vx / -limB;
  // Move stick is vertical-only now that strafe is gone.
  dotMove.style.transform = `translate(0px, ${-yN * STICK_R}px)`;
  dotTurn.style.transform = `translate(${(-wz / limA) * STICK_R}px, 0px)`;
  boxMove.classList.toggle("live", manual && Math.abs(vx) > 0.01);
  boxTurn.classList.toggle("live", manual && Math.abs(wz) > 0.01);

  // Keycap highlighting: each individual key lights only while its own
  // action is active, and only for its own input device (pressed-state
  // snapshots come straight from the controller's sources).
  const sitting = mode === "sitstand" && sitFlag === 1;
  for (const k of ["fwd", "back", "turnl", "turnr"]) {
    keyEls[k].classList.toggle("lit", kbSource.pressed[k]);
  }
  const trick = mode === "roll" || mode === "crouch"; // same R / pad-X slot
  keyEls.roll.classList.toggle("lit", trick && rollSource === "kb");
  keyEls.kickl.classList.toggle("lit", mode === "kickL" && kickSource === "kb");
  keyEls.kickr.classList.toggle("lit", mode === "kickR" && kickSource === "kb");
  keyEls.reset.classList.toggle("lit", performance.now() - resetFlashAt < 400);
  keyEls.ball.classList.toggle("lit", performance.now() - ballFlashAt < 400);
  keyEls.cam.classList.toggle("lit", chaseCam); // steady while chasing
  keyEls.loco?.classList.toggle("lit", loco === "rollers" || locoSwitching);
  keyEls.padX.classList.toggle("lit", trick && rollSource === "pad");
  keyEls.padY.classList.toggle("lit", performance.now() - padYFlashAt < 400);
  keyEls.padRb.classList.toggle("lit", mode === "kickR" && kickSource === "pad");
  keyEls.padLb.classList.toggle("lit", mode === "kickL" && kickSource === "pad");
  keyEls.padSit.classList.toggle("lit", sitting);
  keyEls.padRun.classList.toggle("lit", padSource.pressed.dpadUp);
  keyEls.padRt.classList.toggle("lit", padJaw > 0.3);
  keyEls.padRs.classList.toggle("lit", padOrbitLive); // while deflected
  keyEls.padR3.classList.toggle("lit", chaseCam); // steady while chasing

  drawMinimap();
}

// ── Minimap (90s radar, top-right OSD) ──────────────────────────────────
// Fixed top-down view of the 3x3 m arena. World +X (the spawn facing)
// points UP on the map, so the spawn cell sits bottom-middle; world +Y
// points LEFT (right-handed seen from above). Duck = oriented chevron,
// ball = white dot while spawned, revealed ghosts = faint dots.
const minimapEl = document.getElementById("minimap");
const minimapCtx = minimapEl?.getContext("2d");
const MINIMAP_HZ = 20;
const MINIMAP_ORANGE = "#ff7a2f";
let minimapLastDraw = 0;
// Chevron heading: the minimap keeps its OWN yaw EMA, independent of the
// camera's hysteresis state (chaseHeldYaw only updates while the chase
// cam is attached, and the followed yaw freezes on purpose). The EMA
// averages out the ±14 deg per-step gait wobble while tracking real
// turns continuously; during roll/kick the trunk quaternion tumbles, so
// the last sane value is held (same protection as the camera).
let minimapYaw = 0; // spawn yaw = 0
const MINIMAP_YAW_EASE = 0.15; // per 20 Hz tick, ~0.3 s to settle on a turn
function drawMinimap() {
  if (!minimapCtx) return;
  const now = performance.now();
  if (now - minimapLastDraw < 1000 / MINIMAP_HZ) return;
  minimapLastDraw = now;
  const ctx = minimapCtx;
  const S = minimapEl.width; // square, drawn in device px
  // World (MJCF, Z-up) -> map px: u right = -Y, v down = -X.
  const u = (y) => (0.5 - y / (2 * ARENA_HALF)) * S;
  const v = (x) => (0.5 - x / (2 * ARENA_HALF)) * S;
  ctx.clearRect(0, 0, S, S);
  // Section grid (0.6 m pitch): inner lines only, the CSS border frames it.
  ctx.strokeStyle = "rgba(255, 122, 47, 0.18)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 1; i < 5; i++) {
    const c = (i / 5) * S;
    ctx.moveTo(c, 0); ctx.lineTo(c, S);
    ctx.moveTo(0, c); ctx.lineTo(S, c);
  }
  ctx.stroke();
  const qpos = data.qpos;
  // Ghosts: revealed peers only, faint white dots.
  ctx.fillStyle = "rgba(255, 255, 255, 0.4)";
  for (const g of ghosts?.mapDots() ?? []) {
    ctx.beginPath();
    ctx.arc(u(g.y), v(g.x), S * 0.015, 0, Math.PI * 2);
    ctx.fill();
  }
  // Ball: only while spawned (parked = 50 m away = hidden).
  if (ballActive) {
    const bx = qpos[ballQposAdr], by = qpos[ballQposAdr + 1];
    if (Math.abs(bx) <= ARENA_HALF && Math.abs(by) <= ARENA_HALF) {
      ctx.fillStyle = "rgba(255, 255, 255, 0.92)";
      ctx.beginPath();
      ctx.arc(u(by), v(bx), S * 0.019, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  // Duck: chevron at the trunk position, nose along the smoothed live yaw
  // (frozen during roll/kick where the trunk tumbles).
  if (mode !== "roll" && !isKick()) {
    const rawYaw = Math.atan2(
      2 * (qpos[3] * qpos[6] + qpos[4] * qpos[5]),
      1 - 2 * (qpos[5] * qpos[5] + qpos[6] * qpos[6]),
    );
    minimapYaw = wrapPi(minimapYaw + wrapPi(rawYaw - minimapYaw) * MINIMAP_YAW_EASE);
  }
  ctx.save();
  ctx.translate(u(qpos[1]), v(qpos[0]));
  // Forward (cos yaw, sin yaw) in world -> (-sin yaw, -cos yaw) on the
  // map, i.e. a canvas rotation of -yaw applied to an up-pointing shape.
  ctx.rotate(-minimapYaw);
  const r = S * 0.032;
  ctx.fillStyle = MINIMAP_ORANGE;
  ctx.beginPath();
  ctx.moveTo(0, -r * 1.35); // nose
  ctx.lineTo(r * 0.85, r * 0.95);
  ctx.lineTo(0, r * 0.45); // notched tail = chevron
  ctx.lineTo(-r * 0.85, r * 0.95);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

// ── Right-stick camera orbit (inertia downstream of the controller) ─────
// The gamepad source reports the raw deflection (controller axes.orbitX/Y);
// this step turns it into OrbitControls-compatible motion: rebuild the
// camera-target offset as a spherical, nudge azimuth/elevation, and
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

// Multiplayer ghosts, initialised asynchronously at the end of the module.
let ghosts = null;

let inputPollT = performance.now();
function loop() {
  requestAnimationFrame(loop);
  // dt clamped so a background-tab stall can't slingshot the camera orbit.
  const inputNow = performance.now();
  const dt = Math.min((inputNow - inputPollT) / 1000, 0.05);
  inputPollT = inputNow;
  controller.update(dt);
  padJaw = controller.getAxes().jaw;
  document.body.classList.toggle("pad-connected", padSource.connected);
  document.body.classList.toggle("touch-mode", touchSource.connected);
  // Camera orbit runs every frame while a pad is present (the coasting
  // needs the zero-deflection frames too); without a pad, park the state.
  if (padSource.connected) {
    padOrbitStep(controller.getAxes().orbitX, controller.getAxes().orbitY, dt);
  } else {
    padOrbitLive = false;
    padOrbitVel.az = 0;
    padOrbitVel.el = 0;
  }
  syncRig();
  syncJaw();
  ghosts?.update();
  controls.update();
  updateChaseCam();
  ceremony.drive();
  ball.drive(() => spawnBall({ fromQueue: true }));
  renderStats();
  renderer.render(scene, camera);
}
// Boot complete: the sim/HUD go live immediately. The BIOS readout (if
// the user already waddled in, or when they do) sees bootDone and closes
// with READY. + fade on its own.
bootDone = true;
hudEl.hidden = false;
loop();

// ── Input wiring: arm the controller sources, bind actions to triggers ──
// Key/button mappings live in controls/keyboard.js and controls/gamepad.js;
// this section is pure game-side wiring. Arming the listeners HERE (not at
// source construction) keeps the boot behavior identical to when the raw
// keydown/gamepad handlers attached at this point in the module.
controller.init();

// Keyboard F alternates kicking feet; only advance the alternation on
// kicks that actually launched (triggerKick reports that).
let kbKickFoot = "left";
// Action meta.source ("keyboard"/"gamepad") -> the trigger functions'
// historical source tags, which the HUD keycap lighting keys off.
const srcTag = (source) => (source === "gamepad" ? "pad" : "kb");

controller.on("reset", () => resetSim());
controller.on("spawnBall", ({ source }) => {
  spawnBall();
  if (source === "gamepad") padYFlashAt = performance.now();
  else ballFlashAt = performance.now();
});
controller.on("chaseToggle", () => { chaseCam = !chaseCam; });
controller.on("locoToggle", () => toggleLoco());
controller.on("roll", ({ source }) => triggerRoll(srcTag(source)));
controller.on("kickL", ({ source }) => triggerKick("left", srcTag(source)));
controller.on("kickR", ({ source }) => triggerKick("right", srcTag(source)));
controller.on("alternateKick", ({ source }) => {
  if (triggerKick(kbKickFoot, srcTag(source))) {
    kbKickFoot = kbKickFoot === "left" ? "right" : "left";
  }
});
controller.on("sitToggle", () => {
  if (loco !== "legs") return; // sitting is a legs-only skill
  const sitting = mode === "sitstand" && sitFlag === 1;
  setMode(sitting ? "walk" : "sit");
});
// Pad DpadUp short press: straight back to running (ignored mid-roll /
// mid-crouch: those hand back to walk on their own, and switching on a
// tipped duck would floor it).
controller.on("walk", () => {
  if (mode !== "walk" && mode !== "roll" && mode !== "crouch") setMode("walk");
});
controller.on("quack", () => quackLoud());

// Read-only state label (bottom-left): reflects the active policy,
// switching happens via keyboard/gamepad only.
const modeLabel = document.getElementById("mode-label");

function setMode(next, { force = false } = {}) {
  if (!force && inputLocked) return;
  // No policy switching mid-roll or mid-kick: both end on their own and
  // return to walk - switching now would floor the duck. Sitting is a
  // legs-only skill (the roller stance has no sitstand policy).
  if ((mode === "roll" && rollRun) || (isKick() && kickRun) || (mode === "crouch" && crouchRun)) return;
  if (next === "sit" && loco === "rollers") return;
  clearModeTimers();
  rollRun = null;
  crouchRun = null;
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
  // Same trigger slot in the roller variant fires its own trick.
  if (loco === "rollers") return triggerCrouch(source);
  // Rolls only launch from a standing walk: from a sit (or mid sit/stand
  // hand-over) the roll policy just faceplants the duck.
  if (inputLocked || mode !== "walk" || standTimer) return;
  clearModeTimers();
  rollSource = source;
  mode = "roll";
  sitFlag = 0;
  rollRun = { steps: 0, tipped: false };
  syncButtons();
  stickers?.pop("roll");
}

// Roller-only one-shot: crouch, glide low, stand back up (phase-driven,
// see the crouch constants up top). Reuses the roll's trigger + keycap.
function triggerCrouch(source = "kb") {
  if (inputLocked || mode !== "walk" || locoSwitching) return;
  clearModeTimers();
  rollSource = source;
  mode = "crouch";
  crouchRun = { phase: 0 };
  syncButtons();
  stickers?.pop("roll"); // same WHEE - the crouch-glide is the roller "roll"
}

// One blind kick (the duck can't see any ball - it's a scripted boot),
// left or right leg. Same launch constraints as the roll. Returns whether
// the kick actually launched so the keyboard's foot alternation only
// advances on real kicks.
function triggerKick(foot, source = "kb") {
  // Kicks are legs-only: in roller mode the ball is played by driving.
  if (loco === "rollers") return false;
  if (inputLocked || mode !== "walk" || standTimer) return false;
  clearModeTimers();
  kickSource = source;
  mode = foot === "left" ? "kickL" : "kickR";
  sitFlag = 0;
  kickRun = { steps: 0 };
  syncButtons();
  stickers?.pop("kick");
  return true;
}
// Matrix-style letter scramble: on change every glyph flips through random
// charset entries, then locks to its target left-to-right over ~0.45s.
// One interval for the whole run; monospace keeps the width stable.
const SCRAMBLE_GLYPHS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789#$%&*<>/=+";
let scrambleTimer = null;
function setModeLabel(text) {
  // Guard on the intended target, not the displayed text (which is mid-
  // scramble noise while the interval runs).
  if (modeLabel.dataset.target === text) return;
  modeLabel.dataset.target = text;
  if (scrambleTimer) { clearInterval(scrambleTimer); scrambleTimer = null; }
  const span = modeLabel.firstElementChild;
  const target = text.toUpperCase();
  const n = target.length;
  const DUR = 450; // ms until the last letter locks
  const rnd = () => SCRAMBLE_GLYPHS[(Math.random() * SCRAMBLE_GLYPHS.length) | 0];
  const t0 = performance.now();
  scrambleTimer = setInterval(() => {
    // Letters 0..k-1 are locked; the rest keep boiling.
    const k = Math.floor(((performance.now() - t0) / DUR) * n);
    if (k >= n) {
      clearInterval(scrambleTimer);
      scrambleTimer = null;
      span.textContent = target;
      return;
    }
    let out = target.slice(0, k);
    for (let i = k; i < n; i++) out += rnd();
    span.textContent = out;
  }, 40);
}

function syncButtons() {
  const sitting = mode === "sitstand" && sitFlag === 1;
  setModeLabel(
    mode === "roll" ? "Roll"
    : mode === "crouch" ? "Crouch"
    : isKick() ? "Kick"
    : sitting ? "Sit"
    : loco === "rollers" ? "Drive"
    : "Run",
  );
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
  // model/data are getters: activateLoco swaps them wholesale.
  get model() { return model; },
  get data() { return data; },
  mujoco, camera, controls,
  get mode() { return mode; },
  get sitFlag() { return sitFlag; },
  buildObs, cmd,
  // velCmd is the keyboard source's live command array: writing to it
  // still drives the duck when no other source is active (the controller
  // falls back to it), same as before the controls/ refactor.
  velCmd: kbSource.command, lastAction, resetSim,
  controller, kbSource, padSource,
  spawnBall, triggerKick, triggerRoll, sessions, ort,
  get loco() { return loco; },
  get locoSwitching() { return locoSwitching; },
  toggleLoco, setLoco, ensureRollers,
  triggerCrouch,
  get crouchPhase() { return crouchRun?.phase ?? null; },
  get kickSteps() { return KICK_STEPS; },
  set kickSteps(v) { KICK_STEPS = v; },
  get ballActive() { return ballActive; },
  get ballQposAdr() { return ballQposAdr; },
  get chaseCam() { return chaseCam; },
  set chaseCam(v) { chaseCam = !!v; },
  get camResetActive() { return camResetT0 !== null; },
  get respawnActive() { return ceremony?.respawnActive ?? false; },
  get camPose() {
    return {
      pos: camera.position.toArray(),
      target: controls.target.toArray(),
    };
  },
  get chaseYaw() { return { follow: chaseYawFollow, smooth: chaseYawSmooth, held: chaseHeldYaw, tracking: chaseYawTracking, map: minimapYaw }; },
  padOrbitStep,
  jawOpenNow,
  step: async (n = 1) => { for (let i = 0; i < n; i++) await controlStep(); },
  render: () => { syncRig(); renderer.render(scene, camera); },
  // One full render-loop iteration, for tests driving frames manually.
  frame: () => { syncRig(); syncJaw(); controls.update(); updateChaseCam(); ceremony.drive(); renderStats(); renderer.render(scene, camera); },
  get ghosts() { return ghosts; },
  get inputLocked() { return inputLocked; },
  // Deterministic entrance controls for screenshots/tests: setting values
  // detaches the time-based driver; setFx(1) also runs the FX through its
  // finish path (materials restored) and releases the input lock (same
  // cleanup as the real sequence).
  entrance: {
    start: () => ceremony.startEntrance(),
    setReveal: (floor, wall) => ceremony.setReveal(floor, wall),
    setFx: (p) => ceremony.setFx(p),
  },
};

// ── Multiplayer ghosts (WebRTC, serverless signaling) ───────────────────
// Broadcast this duck's pose and render up to 3 other visitors live as
// translucent ducks. Fire-and-forget: any failure just means no ghosts.
const r3 = (x) => Math.round(x * 1000) / 1000;
try {
  // Ghosts only join once the entrance has fully played: the world (and
  // this duck) must stay hidden until then, translucent peers included.
  await ceremony.entranceFinished;
  const { initGhosts } = await import(signed(`./ghosts.js?v=${SELF_V}`));
  ghosts = await initGhosts({
    scene, rig, cloneRig, setJoint, setJawOpen, applyVariant,
    jointNames: JOINT_NAMES,
    // Ghost rig per locomotion flag: peers in roller mode clone the roller
    // rig once this tab has built it, and fall back to the leg rig until
    // then (their wheels also don't spin - joints aren't broadcast for
    // the passive hinges). Known v1 limitation, documented in the README.
    getRigFor: (l) => (l && locos.rollers ? locos.rollers.rig : locos.legs.rig),
    getLocalState: () => {
      const qpos = data.qpos;
      const j = new Array(NUM_JOINTS);
      for (let i = 0; i < NUM_JOINTS; i++) j[i] = r3(qpos[qposAdr[i]]);
      return {
        p: [r3(qpos[0]), r3(qpos[1]), r3(qpos[2]), r3(qpos[3]), r3(qpos[4]), r3(qpos[5]), r3(qpos[6])],
        j,
        w: r3(jawOpenNow()),
        v: currentVariant,
        l: loco === "rollers" ? 1 : 0,
      };
    },
  });
  // "HI!" sticker when another visitor joins (setter-style registration,
  // same trystero build quirk as onPeerLeave inside ghosts.js).
  if (ghosts.room) ghosts.room.onPeerJoin = () => stickers?.pop("hi");
} catch (e) {
  window.__ghostErr = String((e && e.stack) || e);
  console.warn("ghosts disabled:", e);
}
