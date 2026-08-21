// Shared sim constants, lifted verbatim from the pre-React rl.js.

export const POLICY_DIR = "./policies";
export const POLICIES = {
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
export const CROUCH_PERIOD_S = 5.0;
export const CROUCH_END_PHASE = 0.7;

// Kickable ball: radius and parking spot (far away = hidden by default).
export const BALL_RADIUS = 0.05;
export const BALL_PARK_POS = "50 0 0.05";

// Square arena boxing the play area: static walls at +-ARENA_HALF keep
// the ball (and the duck) inside. Tall enough that neither steps over.
export const ARENA_HALF = 1.5; // inner half-size, m
export const ARENA_WALL_H = 0.25;
export const ARENA_WALL_T = 0.05;
// Section grid: 5 cells across the 3 m arena (ODD, so a true middle
// column/row of cells exists; the lattice is shifted half a cell in the
// shaders so the walls land exactly on section lines).
export const GRID_SECTION = (2 * ARENA_HALF) / 5; // 0.6 m
// Spawn: center of the middle section cell in the SECOND ROW FROM THE
// BACK wall. The duck faces +X (identity freejoint quat, walks toward
// local +X), so "back" is the -X wall: row centers sit at x = -1.2,
// -0.6, 0, 0.6, 1.2 -> second row is -0.6; middle column is y = 0.
// MJCF coordinates (three.js: x -> x, y -> -z).
export const SPAWN_X = -ARENA_HALF + 1.5 * GRID_SECTION; // -0.6
export const SPAWN_Y = 0;
