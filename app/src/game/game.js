// Microduck RL playground core: the REAL trained policies, not a procedural
// waddle. Framework-agnostic port of the pre-React rl.js.
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
//
// Integration contract with the React shell:
//   - bootGame({ scene, camera, renderer }) is called once from inside the
//     R3F canvas; it loads everything, wires inputs and starts the 50 Hz
//     control loop.
//   - frame(dt) is called by R3F's useFrame every animation frame; it does
//     everything the old rAF loop did EXCEPT renderer.render (R3F renders).
//   - UI state flows out through the zustand store (throttled), UI intents
//     flow back in through gameApi.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { signed } from "./signed.js";
import {
  POLICIES, JOINT_NAMES, DEFAULT_POSE, NUM_JOINTS, OBS_SIZE, CMD_SIZE,
  ACTION_SCALE, TIMESTEP, DECIMATION, CTRL_DT,
  VEL_FWD, VEL_BACK, VEL_ANG, RVEL_FWD, RVEL_BACK, RVEL_ANG,
  CROUCH_PERIOD_S, CROUCH_END_PHASE,
  BALL_RADIUS, BALL_PARK_POS, ARENA_HALF, SPAWN_X, SPAWN_Y,
  ARCADE_H, ARCADE_W, ARCADE_D, ARCADE_GAP, ARCADE_WALL_GAP,
} from "./constants.js";
import {
  buildRig, cloneRig, loadKinematics, setJoint, setJawOpen, MODEL_DIR, MESH_VERSION,
} from "./duck.js";
import {
  VARIANTS, VARIANT_NAMES, materialHookFor, DEFAULT_VARIANT, applyVariant, specToHex,
} from "./variants.js";
import { Controller } from "./controls/controller.js";
import { KeyboardSource } from "./controls/keyboard.js";
import { GamepadSource } from "./controls/gamepad.js";
import { TouchSource } from "./controls/touch.js";
import * as fx from "./fx/fx-wireframe.js";
import { createCeremony, CAM_RESET_S } from "./ceremony.js";
import { createBallActor } from "./ball-actor.js";
import { initGhosts } from "./ghosts.js";
import { makeInfiniteGrid, makeArenaWalls } from "./arena.js";
import { createBallVisual } from "./ball-visual.js";
import { useGame, gameApi, bootLine, bootNote, bootHalt } from "../store.js";

// Physics + inference runtimes stay on the CDN, exactly like the pre-Vite
// app: mujoco.js resolves its .wasm sidecar relative to its own URL, and
// onnxruntime fetches its wasm from wasmPaths - neither ever touches the
// bundle. @vite-ignore keeps Rollup's static analysis out of it.
const MUJOCO_URL = "https://cdn.jsdelivr.net/npm/@mujoco/mujoco@3.11.0/mujoco.js";
const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/ort.min.mjs";

// Colour dot shown per variant in the quickbar. Variants can force theirs
// with a `swatch` spec (purple does: its head is warm gray but its
// identity is the purple accents).
const SWATCH_SLOT = { classic: "feet", charcoal: "headDome", purple: "feet", blue: "facePlate" };
export const VARIANT_SWATCHES = Object.fromEntries(
  VARIANT_NAMES.map((name) => {
    const v = VARIANTS[name];
    return [name, specToHex(v.swatch ?? v[SWATCH_SLOT[name] ?? "bodyShell"])];
  }),
);

let bootStarted = false;

export async function bootGame({ scene, camera, renderer }) {
  if (bootStarted) return;
  bootStarted = true;
  try {
    await boot({ scene, camera, renderer });
  } catch (err) {
    console.error("[game] boot failed", err);
    bootHalt(err?.message || String(err));
  }
}

async function boot({ scene, camera, renderer }) {
  const setStore = useGame.setState;
  const store = useGame.getState;

  bootNote("Microduck BIOS v1.0");
  bootLine("MEMORY CHECK")("640K OK");
  bootLine("DUCK FIRMWARE")("PRESENT");

  // Surface async boot failures in the BIOS halt screen. Gated on the boot
  // still being in flight: post-boot async noise (ghost relay hiccups,
  // audio autoplay rejections...) must NOT cue the halt screen.
  const bootGuard = (e, msg) => {
    if (!store().bootDone && !store().bootFailed) bootHalt(msg);
  };
  window.addEventListener("unhandledrejection", (e) => {
    console.error("[game] unhandled rejection", e.reason);
    bootGuard(e, e.reason?.message || String(e.reason));
  });
  window.addEventListener("error", (e) => {
    console.error("[game] window error", e.message);
    bootGuard(e, e.message);
  });

  // Halting at the failure site: a rejected await inside this async boot
  // would otherwise only surface through the caller's catch.
  const traced = (label, p) => {
    const done = bootLine(label);
    return p.then(
      (v) => { done("OK"); return v; },
      (err) => {
        done("FAILED");
        console.error(`[game] ${label} FAILED`, err);
        bootHalt(err?.message || String(err));
        throw err;
      },
    );
  };

  // ── Runtimes (CDN) ──────────────────────────────────────────────────
  const [{ default: loadMujocoFactory }, ort] = await traced(
    "RUNTIME MODULES",
    Promise.all([
      import(/* @vite-ignore */ MUJOCO_URL),
      import(/* @vite-ignore */ ORT_URL),
    ]),
  );
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/";
  ort.env.wasm.numThreads = 1; // static hosting sends no COOP/COEP headers

  // ── MJCF preparation ────────────────────────────────────────────────
  // robot_allcollisions.xml is what infer_policy.py's scene.xml includes:
  // it carries body/shell collision geoms that robot_walk.xml lacks, which
  // the sitstand policy needs (a sit rests the trunk on the ground).
  // Visual meshes are irrelevant to the dynamics: every body carries an
  // explicit <inertial>, and visual geoms have contype=0 conaffinity=0.
  // Stripping them means the MuJoCo VFS only needs the ~10 meshes
  // referenced by collision geoms. Works for both variants.
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
    const ht = 0.05 / 2, hh = 0.25 / 2;
    const off = ARENA_HALF + ht, span = ARENA_HALF + 0.05;
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
    // Arcade cabinet row: one static box covering all three cabinets so
    // the duck and ball can't clip through them. The row sits against the
    // back (-X) wall, centered on y = 0 (three.js z = 0), axis-aligned.
    const rowHalfW = (3 * ARCADE_W + 2 * ARCADE_GAP) / 2;
    doc.querySelector("worldbody").appendChild(
      el("geom", {
        name: "arcade_row", type: "box",
        pos: `${-(ARENA_HALF - ARCADE_WALL_GAP - ARCADE_D / 2)} 0 ${ARCADE_H / 2}`,
        size: `${ARCADE_D / 2} ${rowHalfW} ${ARCADE_H / 2}`,
      }),
    );
    // Kickable ball: a light free sphere (beach-ball feel). MuJoCo has no
    // restitution parameter - the bounce comes from solref damping < 1, and
    // the rolling-friction term makes it come to rest. Appended AFTER the
    // robot body so the trunk freejoint stays first in qpos.
    const ballBody = el("body", { name: "ball", pos: BALL_PARK_POS });
    ballBody.appendChild(el("freejoint", { name: "ball_freejoint" }));
    // condim 6 enables the torsional + rolling friction components; with
    // the default condim 3 a rolling ball never decelerates.
    ballBody.appendChild(el("geom", {
      name: "ball_geom", type: "sphere", size: String(BALL_RADIUS),
      mass: "0.03", friction: "0.4 0.01 0.003", solref: "0.03 0.4", condim: "6",
    }));
    doc.querySelector("worldbody").appendChild(ballBody);
    // STAND keyframe from mjlab's scene_walk.xml. qpos must cover every
    // joint in document order: the 14 actuated hinges take DEFAULT_POSE by
    // name, anything else (the roller variant's passive wheels) starts at
    // zero. The ball's 7 free-joint values MUST be appended or nq won't
    // match; parked 50 m away = effectively absent.
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

  // ── Boot physics + policy in parallel with the render rig ────────────
  const [mujoco, { xml, meshFiles }, k] = await Promise.all([
    traced("MUJOCO WASM", loadMujocoFactory()),
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
  // Always boot on the classic (orange) colourway; the quickbar re-skins live.
  let currentVariant = DEFAULT_VARIANT;
  const rigPromise = (async () => {
    const doneRig = bootLine("RENDER RIG");
    try {
      const builtRig = await buildRig(k, { materialForMesh: materialHookFor(VARIANTS[currentVariant]) });
      doneRig("OK");
      return builtRig;
    } catch (err) {
      doneRig("FAILED");
      bootHalt(err?.message || String(err));
      throw err;
    }
  })();
  // Boot policies with a live [n/5] counter on the BIOS line.
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

  // Addresses resolved once per compiled variant. qpos/qvel/sensordata
  // views are re-read at each use: the WASM heap can grow and detach
  // earlier TypedArray views.
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

  const lastAction = new Float32Array(NUM_JOINTS);
  const obs = new Float32Array(OBS_SIZE);
  const cmd = new Float32Array(CMD_SIZE); // [vx, vy, wz, head(4), body(6)]
  // Input controller: keyboard + gamepad + touch sources merged into one
  // continuous command + discrete action surface, in priority order.
  const kbSource = new KeyboardSource({ getVelocityLimits: () => velLims() });
  const padSource = new GamepadSource({ getVelocityLimits: () => velLims() });
  const touchSource = new TouchSource({ getVelocityLimits: () => velLims() });
  // Keyboard last: it reads zero when idle, so it doubles as the fallback.
  const controller = new Controller({ sources: [padSource, touchSource, kbSource] });
  // Right-stick camera state, read by the telemetry before the camera-orbit
  // section below has evaluated.
  let padOrbitLive = false;
  // Robot input gate: twist commands, mode changes, rolls, kicks and ball
  // spawns all stay inert until the entrance sequence has fully played out.
  let inputLocked = true;
  let ceremony = null;
  let ball = null;
  let stickers = null; // comic popups, currently disabled

  let mode = "walk"; // "walk" | "sitstand" | "roll" | "kickL" | "kickR" | "crouch"
  let sitFlag = 0;
  const isKick = () => mode === "kickL" || mode === "kickR";
  // Local-only kickable ball: false while parked at the keyframe spot
  // (mesh hidden), true once popped in front of the duck.
  let ballActive = false;

  // The twist the policy actually receives. Mid-roll every movement input
  // is ignored (zero twist) until the roll hands back to walk on its own.
  const ZERO_CMD = new Float32Array(3);
  function effectiveCmd() {
    if (inputLocked || mode === "roll" || mode === "crouch" || isKick() || postKickLock > 0)
      return ZERO_CMD;
    return controller.getCommand();
  }
  let rollRun = null;
  let crouchRun = null;
  let kickRun = null;
  let KICK_STEPS = 25;
  // Post-kick grace: keep commands zeroed for a beat after the kick window
  // hands back to walk. Step-counted like everything else.
  const POST_KICK_LOCK_STEPS = 20; // 0.4 s at 50 Hz
  let postKickLock = 0;

  // Pending mode-transition timers (sit hand-over, stand-up hand-back).
  let sitTimer = null;
  let standTimer = null;
  let fallenSince = null;
  function clearModeTimers() {
    clearTimeout(sitTimer); sitTimer = null;
    clearTimeout(standTimer); standTimer = null;
  }

  function resetSim() {
    // Single reset path: Space, fall-kill, failed roll, loco switch.
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
    syncButtons();
    ceremony?.playRespawn();
  }
  resetSim();

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

  // Pop / respawn the ball ~0.35 m in front of the duck, with a small
  // random heading + distance jitter. If the ball is already on screen,
  // peel it away first (reverse scan) and pop the new one when that
  // finishes - same appear/disappear pair as the duck's wireframe ceremony.
  function spawnBall(opts = {}) {
    if (inputLocked && !opts.fromQueue) return;
    if (!ball) return;
    if (ball.visual !== "hidden") {
      ball.queueRespawn();
      ball.despawn({ parkPhysics: parkBallPhysics });
      return;
    }
    const qpos = data.qpos, qvel = data.qvel;
    const yaw = Math.atan2(
      2 * (qpos[3] * qpos[6] + qpos[4] * qpos[5]),
      1 - 2 * (qpos[5] * qpos[5] + qpos[6] * qpos[6]),
    );
    const heading = yaw + (Math.random() - 0.5) * 0.7;
    const dist = 0.35 + (Math.random() - 0.5) * 0.1;
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

  // ── Observation ─────────────────────────────────────────────────────
  const _q = new THREE.Quaternion();
  const _g = new THREE.Vector3();

  function buildObs() {
    const qpos = data.qpos, qvel = data.qvel, sens = data.sensordata;
    let i = 0;
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

  // ── Control loop (50 Hz, async because ONNX inference is async) ──────
  let ctrlHz = 0;

  // Dead pose: walk/sitstand have no get-up skill, so a kill here is just
  // a resetSim. "fallen" = trunk tilted past ~60 deg or sunk below the
  // floor. NaN/Inf is a solver explosion: no grace, reset on the spot.
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

    // Ball respawn watchdog: outside the arena bounds means "escaped
    // through a solver glitch", bring it back near the duck.
    if (ballActive) {
      const q = data.qpos;
      const escaped =
        Math.abs(q[ballQposAdr]) > ARENA_HALF + 0.1 ||
        Math.abs(q[ballQposAdr + 1]) > ARENA_HALF + 0.1;
      if (escaped) spawnBall();
    }

    if (postKickLock > 0 && mode === "walk") postKickLock--;

    // One-shot kick: fixed 0.5 s window like the robot runtime, then
    // straight back to walking. lastAction is NOT zeroed on either swap.
    if (isKick() && kickRun) {
      kickRun.steps++;
      if (kickRun.steps >= KICK_STEPS) {
        kickRun = null;
        mode = "walk";
        postKickLock = POST_KICK_LOCK_STEPS;
        syncButtons();
      }
    }

    // Crouch-glide one-shot: advance the trained phase clock and hand back
    // to the drive policy at the runtime's cycle end.
    if (mode === "crouch" && crouchRun) {
      crouchRun.phase += CTRL_DT / CROUCH_PERIOD_S;
      if (crouchRun.phase >= CROUCH_END_PHASE) {
        crouchRun = null;
        mode = "walk";
        syncButtons();
      }
    }

    // One-shot roll, step-counted like the robot runtime: hand back to
    // walking once the trunk has tipped over and is upright again, or
    // after a hard window if the roll never initiated.
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
        // Timed out mid-roll: don't hand a tipped duck to the walking
        // policy (it has no get-up skill).
        if (!upright) resetSim();
        syncButtons();
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

  // ── Scene wiring (grid, walls, rig, ball, arcade row) ────────────────
  // The grid/walls carry ceremony-driven uReveal uniforms and per-frame
  // focus updates, so the game owns them; lights and environment live in
  // the R3F layer.
  const grid = makeInfiniteGrid();
  scene.add(grid);
  const { wallMats, wallMeshes } = makeArenaWalls();
  for (const m of wallMeshes) scene.add(m);

  let rig = await rigPromise;
  scene.add(rig.placer);
  let trunkGroup = rig.bodies.get("trunk_base");
  locos.legs = {
    model, data, rig, trunkGroup,
    qposAdr, dofAdr, gyroAdr, trunkId, standKeyId, ballQposAdr, ballDofAdr, extraJoints,
  };

  // ── Locomotion variant switching (legs <-> rollers) ──────────────────
  // The roller stack (XML + 5 extra meshes + kinematics + 2 ONNX policies)
  // is lazy-loaded on the first switch, then kept resident.
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
    setStore({ loco: name });
    resetSim();
  }

  let locoSwitching = false;
  async function setLoco(name, { force = false } = {}) {
    if (name !== "legs" && name !== "rollers") return;
    if (loco === name || locoSwitching) return;
    if (!force && (inputLocked || rollRun || kickRun || crouchRun || standTimer)) return;
    locoSwitching = true;
    setStore({ locoSwitching: true });
    try {
      if (name === "rollers" && !locos.rollers) {
        setStore({ rollersLoading: true });
        await ensureRollers();
      }
      activateLoco(name);
    } catch (e) {
      rollersLoading = null;
      console.error("[game] roller switch failed", e);
    } finally {
      setStore({ rollersLoading: false, locoSwitching: false });
      locoSwitching = false;
    }
  }

  async function toggleLoco() {
    const next = loco === "legs" ? "rollers" : "legs";
    setStore({ locoWant: next });
    await setLoco(next);
  }

  // Quickbar loco intent: reconcile locoWant -> actual, retrying until the
  // game allows the switch (mid-roll, respawn ceremony, ...). Replaces the
  // old index.html reconciler that polled window.rl.
  let locoReconciler = null;
  function reconcileLoco() {
    const want = store().locoWant;
    if (want === loco) {
      if (locoReconciler) { clearInterval(locoReconciler); locoReconciler = null; }
      return;
    }
    if (want === "rollers") ensureRollers().catch(() => {});
    if (!locoSwitching) setLoco(want);
    locoReconciler ??= setInterval(reconcileLoco, 250);
  }
  useGame.subscribe((s) => s.locoWant, reconcileLoco);

  // ── Cutscenes (entrance + respawn) ──────────────────────────────────
  ceremony = createCeremony({
    THREE, scene, camera, renderer, fx,
    getRig: () => rig,
    grid, wallMats,
    syncRig, startCameraReset,
    setLocked: (v) => {
      inputLocked = v;
      controller.setLocked(v);
      // A ball is always in play: pop one the moment the entrance or a
      // respawn ceremony hands control back.
      if (!v && ball && !ballActive) spawnBall({ fromQueue: true });
    },
    flashReset: () => {},
  });

  const { group: ballGroup, mesh: ballMesh } = createBallVisual(renderer);
  scene.add(ballGroup);
  ball = createBallActor({
    THREE, scene, camera, renderer, fxModule: fx, mesh: ballMesh, group: ballGroup,
  });

  // ── Arcade cabinets (wall dressing + static collision box) ──────────
  // assets/props/arcade.glb carries two meshes: the ~9k-tri render mesh
  // and a 500-tri "arcade_wire" stand-in used only by the hologram pass
  // (same fxWireGeometry escape hatch as the ball). Three clones line up
  // against the back (-X) wall, backs to the wall, screens facing the
  // arena; each materializes with the duck's ceremony, staggered, and
  // replays on every respawn. Physics-side, buildPhysicsXml plants one
  // static box covering the whole row (constants ARCADE_*).
  let arcadeGroups = null;
  try {
    const gltf = await new GLTFLoader().loadAsync(
      signed(`./assets/props/arcade.glb?v=1`),
    );
    const proto = gltf.scene;
    const wire = proto.getObjectByName("arcade_wire");
    wire.removeFromParent();
    const box3 = new THREE.Box3().setFromObject(proto);
    const scale = ARCADE_H / (box3.max.y - box3.min.y);
    proto.scale.setScalar(scale);
    proto.position.y = -box3.min.y * scale;
    arcadeGroups = [];
    for (let i = 0; i < 3; i++) {
      // clone(true) shares geometry/materials; userData doesn't survive
      // Object3D.copy (JSON round-trip), so tag the wire geometry after.
      const cab = proto.clone(true);
      cab.traverse((o) => {
        if (o.isMesh) o.userData.fxWireGeometry = wire.geometry;
      });
      const group = new THREE.Group();
      group.add(cab);
      // Row centered on the back wall's middle, one cabinet-width apart,
      // back faces almost touching the wall. The GLB fronts +Z; the row
      // must front +X (toward the spawn), hence the +90 deg yaw.
      group.position.set(
        -(ARENA_HALF - ARCADE_D / 2 - ARCADE_WALL_GAP),
        0,
        (i - 1) * (ARCADE_W + ARCADE_GAP),
      );
      group.rotation.y = Math.PI / 2;
      scene.add(group);
      arcadeGroups.push(group);
      const cabFx = fx.createWireframeFx();
      cabFx.init({ THREE, scene, root: group, camera, renderer, hidden: true });
      ceremony.addPropFx(cabFx, 0.25 + i * 0.12);
    }
  } catch (err) {
    // Decorative only: a missing/broken GLB must never halt the boot.
    console.warn("[game] arcade props disabled:", err);
  }

  // ── Camera: orbit controls + chase cam + reset glide ─────────────────
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
  // OrbitControls' own spherical bookkeeping.
  let chaseCam = true;
  const CHASE_PITCH = 0.42; // rad above horizontal, keeps the floor in view
  const CHASE_EASE = 0.05;
  const _chasePos = new THREE.Vector3();
  const _chaseDir = new THREE.Vector3();
  // During one-shot rolls and kicks the trunk tumbles: hold the last
  // healthy yaw for the whole one-shot.
  let chaseHeldYaw = 0;
  // Heading hysteresis (Schmitt trigger): the walking gait wiggles the
  // trunk yaw ~±14 deg per step; two-layer EMA + engage/release thresholds
  // keep the camera steady while walking straight but responsive on turns.
  let chaseYawSmooth = 0;
  let chaseYawFollow = 0;
  let chaseYawTracking = false;
  const CHASE_YAW_SMOOTH_EASE = 0.04;
  const CHASE_YAW_ENGAGE = 0.17;
  const CHASE_YAW_RELEASE = 0.03;
  const CHASE_YAW_EASE = 0.10;
  const CHASE_YAW_EASE_TURN = 0.5;
  const wrapPi = (a) => Math.atan2(Math.sin(a), Math.cos(a));
  function updateChaseCam() {
    // Reset glide: one clean tween from wherever the camera is back to the
    // home framing. Runs instead of the chase logic and hands control back
    // to it on landing.
    if (camResetT0 !== null) {
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
    // "turning" reads the raw per-source wz commands (not the locked/merged
    // view) so an intentional turn engages on the first frame.
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
    // during large swings.
    _chaseDir.copy(camera.position).sub(controls.target);
    const len = _chaseDir.length();
    if (len > 1e-6) camera.position.copy(controls.target).addScaledVector(_chaseDir, dist / len);
    camera.lookAt(controls.target);
  }
  renderer.domElement.addEventListener("pointerdown", () => { chaseCam = false; });

  // Camera reset glide (owned by the respawn ceremony): back to the
  // page-load framing - the chase cam's ideal point behind the duck's
  // spawn heading, at the boot orbit distance.
  const CAM_HOME_DIST = camera.position.distanceTo(controls.target);
  let camResetT0 = null;
  const _camFrom = new THREE.Vector3(), _camTo = new THREE.Vector3();
  const _tgtFrom = new THREE.Vector3(), _tgtTo = new THREE.Vector3();
  function startCameraReset() {
    const qpos = data.qpos;
    const yaw = Math.atan2(
      2 * (qpos[3] * qpos[6] + qpos[4] * qpos[5]),
      1 - 2 * (qpos[5] * qpos[5] + qpos[6] * qpos[6]),
    );
    chaseHeldYaw = yaw;
    chaseYawSmooth = yaw;
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
  useGame.subscribe(
    (s) => s.menuOpen,
    (open) => {
      if (!ceremony.entranceDone) return;
      if (open) setInputLock(true);
      else if (!ceremony.respawnActive) setInputLock(false);
    },
  );

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
    // Ball: live follows qpos; ghost freeze is owned by the ball actor.
    if (ball) ball.sync(qpos, ballQposAdr, ballActive);
    // Follow cam: ease the orbit target toward the trunk and translate the
    // camera by the same delta, so the camera-to-duck distance and viewing
    // angle stay constant while the duck walks. Paused while the reset
    // glide owns the camera.
    if (camResetT0 === null) {
      _target.set(qpos[0], qpos[2], -qpos[1]);
      _follow.copy(_target).sub(controls.target);
      // Horizontal follow at the usual rate; vertical much slower so the
      // per-step gait bob doesn't nod the frame.
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

  // ── Quack: jaw + chirp ────────────────────────────────────────────────
  // The jaw isn't a MuJoCo joint (duck.js re-creates the hinge in JS), so
  // this is purely cosmetic and can't upset the policy. Voice banks from
  // the robot runtime: each colourway gets its own bank and every quack
  // draws a random chirp take from it.
  const QUACK_MS = 480;
  let quackAt = -Infinity;
  let padJaw = 0;
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

  // ── Telemetry (throttled into the store) ─────────────────────────────
  // FPS EMA is per-frame; the store write is 4 Hz so React re-renders
  // stay far away from frame rate. Odometer integrates horizontal trunk
  // travel; teleport-sized jumps (resets, loco swaps) don't count.
  let fpsEma = 60;
  let fpsLastT = performance.now();
  let odoM = 0;
  let odoX = null, odoY = null;
  let telemetryLastPush = 0;
  function renderTelemetry() {
    const now = performance.now();
    const dtF = (now - fpsLastT) / 1000;
    fpsLastT = now;
    if (dtF > 0 && dtF < 0.5) fpsEma += (1 / dtF - fpsEma) * 0.05;
    const stepD = (odoX === null) ? 0 : Math.hypot(data.qpos[0] - odoX, data.qpos[1] - odoY);
    if (stepD < 0.05) odoM += stepD; // plausible per-frame travel only
    odoX = data.qpos[0];
    odoY = data.qpos[1];
    if (now - telemetryLastPush < 250) return;
    telemetryLastPush = now;
    setStore({
      telemetry: {
        fps: Math.round(fpsEma),
        ctrlHz: Math.round(ctrlHz),
        speed: Math.hypot(data.qvel[0], data.qvel[1]),
        odo: odoM,
        peers: ghosts?.peerCount() ?? 0,
      },
    });
  }

  // ── Right-stick camera orbit (inertia downstream of the controller) ──
  // The stick steers an angular VELOCITY that eases toward the stick's
  // target rate, so pushing ramps up gently and releasing coasts to a stop
  // over ~0.3 s. Vertical is flight-style inverted.
  const PAD_ORBIT_SPEED = 2.4; // rad/s at full deflection
  const PAD_ORBIT_SMOOTH = 8; // 1/s response rate (~95% in 0.37 s)
  const padOrbitVel = { az: 0, el: 0 };
  const _padSph = new THREE.Spherical();
  const _padOff = new THREE.Vector3();
  function padOrbitStep(rx, ry, dt) {
    padOrbitLive = rx !== 0 || ry !== 0;
    if (padOrbitLive) chaseCam = false; // detach, same as a mouse grab
    const k = 1 - Math.exp(-PAD_ORBIT_SMOOTH * dt);
    padOrbitVel.az += (rx * PAD_ORBIT_SPEED - padOrbitVel.az) * k;
    padOrbitVel.el += (-ry * PAD_ORBIT_SPEED * 0.75 - padOrbitVel.el) * k;
    if (chaseCam) { padOrbitVel.az = 0; padOrbitVel.el = 0; return; }
    if (Math.abs(padOrbitVel.az) < 1e-3 && Math.abs(padOrbitVel.el) < 1e-3) return;
    _padOff.copy(camera.position).sub(controls.target);
    _padSph.setFromVector3(_padOff);
    _padSph.theta -= padOrbitVel.az * dt;
    _padSph.phi += padOrbitVel.el * dt;
    _padSph.phi = Math.min(controls.maxPolarAngle, Math.max(0.08, _padSph.phi));
    _padSph.makeSafe();
    camera.position.setFromSpherical(_padSph).add(controls.target);
    camera.lookAt(controls.target);
  }

  // Multiplayer ghosts, initialised asynchronously at the end of the boot.
  let ghosts = null;

  // ── Per-frame drive, called by R3F's useFrame ────────────────────────
  let padWasConnected = null;
  let touchWasConnected = null;
  function frame(dt) {
    controller.update(dt);
    padJaw = controller.getAxes().jaw;
    if (padSource.connected !== padWasConnected) {
      padWasConnected = padSource.connected;
      setStore({ padConnected: padSource.connected });
    }
    if (touchSource.connected !== touchWasConnected) {
      touchWasConnected = touchSource.connected;
      setStore({ touchMode: touchSource.connected });
    }
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
    renderTelemetry();
  }

  // ── Input wiring: arm the controller sources, bind actions ───────────
  controller.init();

  // Keyboard F alternates kicking feet; only advance the alternation on
  // kicks that actually launched (triggerKick reports that).
  let kbKickFoot = "left";
  const srcTag = (source) => (source === "gamepad" ? "pad" : "kb");

  controller.on("reset", () => resetSim());
  controller.on("spawnBall", () => spawnBall());
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
  // mid-crouch: those hand back to walk on their own).
  controller.on("walk", () => {
    if (mode !== "walk" && mode !== "roll" && mode !== "crouch") setMode("walk");
  });
  controller.on("quack", () => quackLoud());

  function setMode(next, { force = false } = {}) {
    if (!force && inputLocked) return;
    // No policy switching mid-roll or mid-kick: both end on their own and
    // return to walk - switching now would floor the duck.
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
      // knocks the duck over.
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

  // One roll, then straight back to running. lastAction is deliberately
  // NOT zeroed: the runtime keeps one continuous action history across
  // policy switches, and the roll initiates more reliably mid-gait.
  function triggerRoll(source = "kb") {
    if (loco === "rollers") return triggerCrouch(source);
    if (inputLocked || mode !== "walk" || standTimer) return;
    clearModeTimers();
    mode = "roll";
    sitFlag = 0;
    rollRun = { steps: 0, tipped: false };
    syncButtons();
    stickers?.pop("roll");
  }

  // Roller-only one-shot: crouch, glide low, stand back up (phase-driven).
  function triggerCrouch(source = "kb") {
    if (inputLocked || mode !== "walk" || locoSwitching) return;
    clearModeTimers();
    mode = "crouch";
    crouchRun = { phase: 0 };
    syncButtons();
    stickers?.pop("roll");
  }

  // One blind kick (the duck can't see any ball - it's a scripted boot).
  // Returns whether the kick actually launched so the keyboard's foot
  // alternation only advances on real kicks.
  function triggerKick(foot, source = "kb") {
    if (loco === "rollers") return false;
    if (inputLocked || mode !== "walk" || standTimer) return false;
    clearModeTimers();
    mode = foot === "left" ? "kickL" : "kickR";
    sitFlag = 0;
    kickRun = { steps: 0 };
    syncButtons();
    stickers?.pop("kick");
    return true;
  }

  function syncButtons() {
    const sitting = mode === "sitstand" && sitFlag === 1;
    const label =
      mode === "roll" ? "Roll"
      : mode === "crouch" ? "Crouch"
      : isKick() ? "Kick"
      : sitting ? "Sit"
      : loco === "rollers" ? "Drive"
      : "Run";
    if (store().modeLabel !== label) setStore({ modeLabel: label });
    if (store().ballActive !== ballActive) setStore({ ballActive });
  }

  // ── Public surface for the React UI ──────────────────────────────────
  Object.assign(gameApi, {
    frame,
    setVariant: (name) => {
      if (!VARIANTS[name] || name === currentVariant) return;
      currentVariant = name;
      applyVariant(rig, name);
      setStore({ variant: name });
    },
    requestLoco: (name) => {
      if (name !== "legs" && name !== "rollers") return;
      setStore({ locoWant: name });
      reconcileLoco();
    },
    resetSim,
    spawnBall: () => spawnBall(),
    startEntrance: () => ceremony.startEntrance(),
  });

  // Deterministic hooks for automated verification (rAF pauses in
  // background tabs, and the control loop is async).
  window.rl = {
    get model() { return model; },
    get data() { return data; },
    mujoco, camera, controls,
    get mode() { return mode; },
    get sitFlag() { return sitFlag; },
    buildObs, cmd,
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
    get arcades() { return arcadeGroups; },
    get camResetActive() { return camResetT0 !== null; },
    get respawnActive() { return ceremony?.respawnActive ?? false; },
    get camPose() {
      return {
        pos: camera.position.toArray(),
        target: controls.target.toArray(),
      };
    },
    get chaseYaw() { return { follow: chaseYawFollow, smooth: chaseYawSmooth, held: chaseHeldYaw, tracking: chaseYawTracking }; },
    padOrbitStep,
    jawOpenNow,
    step: async (n = 1) => { for (let i = 0; i < n; i++) await controlStep(); },
    render: () => { syncRig(); renderer.render(scene, camera); },
    frame: (dt = 1 / 60) => frame(dt),
    get ghosts() { return ghosts; },
    get inputLocked() { return inputLocked; },
    entrance: {
      start: () => ceremony.startEntrance(),
      setReveal: (floor, wall) => ceremony.setReveal(floor, wall),
      setFx: (p) => ceremony.setFx(p),
    },
  };

  // Boot complete: the sim/HUD go live immediately. The BIOS readout (if
  // the user already waddled in, or when they do) sees bootDone and closes
  // with READY. + fade on its own.
  setStore({ bootDone: true });

  // ── Multiplayer ghosts (WebRTC, serverless signaling) ────────────────
  // Broadcast this duck's pose and render up to 3 other visitors live as
  // translucent ducks. Fire-and-forget: any failure just means no ghosts.
  const r3 = (x) => Math.round(x * 1000) / 1000;
  try {
    // Ghosts only join once the entrance has fully played: the world (and
    // this duck) must stay hidden until then, translucent peers included.
    await ceremony.entranceFinished;
    ghosts = await initGhosts({
      scene, rig, cloneRig, setJoint, setJawOpen, applyVariant,
      jointNames: JOINT_NAMES,
      // Ghost rig per locomotion flag: peers in roller mode clone the
      // roller rig once this tab has built it, and fall back to the leg
      // rig until then. Known v1 limitation, documented in the README.
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
    if (ghosts.room) ghosts.room.onPeerJoin = () => stickers?.pop("hi");
  } catch (e) {
    window.__ghostErr = String((e && e.stack) || e);
    console.warn("ghosts disabled:", e);
  }
}
