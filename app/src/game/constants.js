// Shared sim constants, lifted verbatim from the pre-React rl.js.

export const POLICY_DIR = "./policies";

// DuckLab can open an immutable Policy Bench ONNX artifact directly in the
// colorful browser arena.  The default Pollen policies remain untouched when
// these query parameters are absent.
const PREVIEW_PARAMS = new URLSearchParams(globalThis.location?.search ?? "");
const previewNumber = (name, fallback, min, max) => {
  if (!PREVIEW_PARAMS.has(name)) return fallback;
  const value = Number(PREVIEW_PARAMS.get(name));
  return Number.isFinite(value) && value >= min && value <= max ? value : fallback;
};
export const PREVIEW_POLICY = PREVIEW_PARAMS.get("preview_policy") || "";
export const PREVIEW_SLOT = ["walk", "drive", "crouch"].includes(PREVIEW_PARAMS.get("preview_slot"))
  ? PREVIEW_PARAMS.get("preview_slot")
  : "";
export const PREVIEW_LOCO = PREVIEW_PARAMS.get("preview_loco") === "rollers" ? "rollers" : "legs";
export const PREVIEW_LABEL = (PREVIEW_PARAMS.get("preview_label") || "").slice(0, 32);
export const SPEED_TEST_MODE = PREVIEW_PARAMS.get("speed_test") === "1";
export const SPEED_TEST_DISTANCE_FT = previewNumber("speed_test_distance_ft", 100, 10, 300);
export const SPEED_TEST_DISTANCE_M = SPEED_TEST_DISTANCE_FT * 0.3048;
// A race policy receives yaw-rate command but has no absolute course heading
// or lateral-position observation.  The line controller closes that missing
// outer loop, just as the operator currently does by holding left.  It is on
// for speed-test previews, explicitly measurable, and can be disabled with
// `auto_line=0` for raw-policy comparison.
export const AUTO_LINE_HOLD = SPEED_TEST_MODE && PREVIEW_PARAMS.get("auto_line") !== "0";
export const LINE_HOLD_YAW_KP = previewNumber("line_yaw_kp", 0.55, 0.0, 3.0);
export const LINE_HOLD_LATERAL_KP = previewNumber("line_lateral_kp", 0.10, 0.0, 1.0);
export const LINE_HOLD_YAW_KD = previewNumber("line_yaw_kd", 0.08, 0.0, 1.0);
export const LINE_HOLD_MAX_WZ = previewNumber("line_max_wz", 0.18, 0.01, 0.30);
// A fixed roller asymmetry can require a small corrective turn while the gait
// first loads.  This deliberately fades over the first metres; it is surfaced
// in the speed-test URL rather than pretending the raw policy is symmetric.
export const LINE_LAUNCH_BIAS_WZ = previewNumber("line_launch_bias_wz", -0.06, -0.30, 0.30);
// Short-lived local dashboard credential used only when the operator
// explicitly saves a state/action demonstration from the arena.
export const CAPTURE_TOKEN = PREVIEW_PARAMS.get("capture_token") ||
  globalThis.__DUCKLAB_CAPTURE_TOKEN__ || "";
export const POLICIES = {
  walk: PREVIEW_POLICY && PREVIEW_SLOT === "walk"
    ? PREVIEW_POLICY
    : `${POLICY_DIR}/BEST_alpha_walking.onnx`,
  sitstand: `${POLICY_DIR}/BEST_alpha_sitstand.onnx`,
  roll: `${POLICY_DIR}/roulade.onnx`,
  // Blind one-shot kicks (the operator aims the robot, no ball in obs):
  // the runtime swaps these in for a 0.5 s window, commands zeroed.
  kickL: `${POLICY_DIR}/ball_kick_left.onnx`,
  kickR: `${POLICY_DIR}/ball_kick_right.onnx`,
  // One-shot ground pick (runtime A button): peck the ground and stand
  // back up, phase-driven via [cos, sin, 0] in the command vel slots.
  groundpick: `${POLICY_DIR}/alpha_ground_pick.onnx`,
  // Get-up policy for the automatic fall recovery (runtime --fall-detect):
  // same 61D obs layout as the other alpha policies, commands all zeroed.
  stand: `${POLICY_DIR}/BEST_alpha_stand.onnx`,
  // Roller variant (lazy-loaded on first switch, never at boot):
  // drive = velocity-tracking skating, crouch = one-shot crouch-glide
  // driven by a phase encoding in the command slots (ground-pick style).
  drive: PREVIEW_POLICY && PREVIEW_SLOT === "drive"
    ? PREVIEW_POLICY
    : `${POLICY_DIR}/BEST_roller.onnx`,
  crouch: PREVIEW_POLICY && PREVIEW_SLOT === "crouch"
    ? PREVIEW_POLICY
    : `${POLICY_DIR}/BEST_roller_crouch.onnx`,
};

// From the ONNX metadata (identical for all alpha policies) and the STAND
// keyframe in mjlab's scene_walk.xml. Order matches the actuators in
// the MJCF.
export const JOINT_NAMES = [
  "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
  "neck_pitch", "head_pitch", "head_yaw", "head_roll",
  "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
];
export const DEFAULT_POSE = new Float32Array([
  0, -0.08726646259971647, -0.457924, -0.004940, 0.452984,
  0.3490658503988659, 0.3490658503988659, 0, 0,
  0, 0.08726646259971647, 0.457924, 0.004940, -0.452984,
]);
export const NUM_JOINTS = 14;
export const OBS_SIZE = 61;
export const CMD_SIZE = 13;
export const ACTION_SCALE = 1.0;
export const TIMESTEP = 0.005;
export const DECIMATION = 4;
export const CTRL_DT = TIMESTEP * DECIMATION; // 50 Hz

// DuckLab standardized race physics. The headless scorer derives the torque cap
// from the XL330 M6 model at 1.75 A (kt=0.36601349688984386 N.m/A) and applies
// the same bearing drag to all four passive wheels. Keep the browser preview
// on these exact values so an arena mph result is comparable to Policy Bench.
// This is a documented DuckLab benchmark profile, not an endorsement or an
// official benchmark published by Pollen Robotics.
export const ROLLER_CURRENT_LIMIT_A = 1.75;
export const ROLLER_TORQUE_LIMIT_NM = 0.6405236195572268;
export const ROLLER_WHEEL_FRICTIONLOSS = 0.003;
export const RACE_EFFORT_COMMAND_MPS = 0.8;

// Velocity command limits, same as infer_policy.py's keyboard mapping.
// No strafe input anymore: the lateral cmd slot stays zeroed for the obs.
export const VEL_FWD = 0.25, VEL_BACK = -0.2, VEL_ANG = 1.0;
// Roller mode limits, from the runtime's roller branch: asymmetric vx
// (0.6 push / 0.5 brake), no lateral. The real runtime launches rollers
// with --max-angular-vel 0.3: faster commanded turns tip the robot over,
// so the playground clamps wz the same way.
export const RVEL_FWD = 0.6, RVEL_BACK = -0.5, RVEL_ANG = 0.3;
// Crouch-glide one-shot: command = [cos(2pi*phase), sin(2pi*phase), 0],
// phase advancing at 1/CROUCH_PERIOD_S per second and the cycle exiting
// at 0.7 - exactly the runtime's ground-pick slot the policy was trained
// against (mjlab CROUCH_PERIOD = 5.0, cycle end 0.7 => 3.5 s gesture).
export const CROUCH_PERIOD_S = previewNumber("preview_period", 5.0, 0.25, 30.0);
export const CROUCH_END_PHASE = previewNumber("preview_end", 0.7, 0.05, 1.0);
// Ground-pick one-shot (legs): same phase encoding, from the runtime's
// defaults (--ground-pick-period 4.0, cycle exiting at 0.7 => ~2.8 s
// gesture, action scale and kP untouched at their 1.0 defaults).
export const GROUND_PICK_PERIOD_S = 4.0;
export const GROUND_PICK_END_PHASE = 0.7;

// Kickable ball: radius and parking spot (far away = hidden by default).
export const BALL_RADIUS = 0.05;
export const BALL_PARK_POS = "50 0 0.05";

// Policy previews open as a long, flat straight-line speed test. The spawn is
// at -0.4 * ARENA_HALF and the forward wall at +ARENA_HALF, so 0.9x the test
// length yields roughly 25% extra runway after the measured finish.
// Ordinary factory play keeps the original compact room.
export const ARENA_HALF = SPEED_TEST_MODE
  ? Math.max(15.24, SPEED_TEST_DISTANCE_M * 0.9)
  : 1.5;
export const ARENA_WALL_H = 0.25;
export const ARENA_WALL_T = 0.05;
// Section grid: 5 cells across the arena (ODD, so a true middle
// column/row of cells exists; the lattice is shifted half a cell in the
// shaders so the walls land exactly on section lines).
export const GRID_SECTION = (2 * ARENA_HALF) / 5; // 0.6 m
// Arcade cabinet row: three cabinets side by side against the front (+X)
// wall, screens facing the arena. Consumed by the prop library (props.js
// "arcade" def - currently benched, enabled: false) for both the clone
// placements and the row's single static collision box. Proportions
// measured from the GLB's natural size (0.524 x 0.587 x 1.0 m, w x d x h).
// Slightly surreal scale: ~2.8x the 0.25 m duck (a real 1.73 m cabinet
// felt overwhelming in the 3 m arena). Big enough to read as oversized
// furniture, small enough that the row (~1.15 m wide, 0.41 m deep)
// leaves the play area open.
export const ARCADE_H = 0.7; // target height, m
export const ARCADE_W = ARCADE_H * 0.524; // footprint width, ~0.22 m
export const ARCADE_D = ARCADE_H * 0.5876; // footprint depth, ~0.247 m
export const ARCADE_GAP = 0.02; // gap between neighbouring cabinets
export const ARCADE_WALL_GAP = 0.01; // clearance between backs and the wall

// Relief (prototype): gentle cosine bumps baked into the level itself.
// One shared analytic height function drives BOTH the physics (a MuJoCo
// heightfield covering the arena) and the visuals (vertex displacement
// of the grid floor shader), so the grid genuinely deforms instead of
// boxes popping out of it. Entries are [cx, cy, height, radius] in MJCF
// coords; slopes stay mild (max grade ~20%) so the blind walking policy
// has a chance, and the spawn cell + arcade row stay flat.
export const RELIEF_BUMPS = [
  [0.3, -0.9, 0.06, 0.55],
  [-0.9, 0.75, 0.05, 0.5],
  [0.6, 0.9, 0.07, 0.55],
];
export const RELIEF_HMAX = 0.07; // tallest bump, scales the hfield z-size
export const RELIEF_GRID = 65; // hfield rows/cols over the 3x3 m arena
export const RELIEF_SINK = 0.003; // hfield geom buried below the floor
export const RELIEF_RATE = 0.6; // raise/sink rate, scale units per second

// Spawn: center of the middle section cell in the SECOND ROW FROM THE
// BACK wall. The duck faces +X (identity freejoint quat, walks toward
// local +X); the second of five cells leaves a long straight runway.
// MJCF coordinates (three.js: x -> x, y -> -z).
export const SPAWN_X = -ARENA_HALF + 1.5 * GRID_SECTION; // -0.6
export const SPAWN_Y = 0;
