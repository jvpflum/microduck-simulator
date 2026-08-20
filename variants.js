// Colour variants of the real Microduck robots (see the reference photo of
// the four printed units). Each variant fills the same semantic slots; the
// mesh-filename -> slot mapping lives in meshMaterialsFor(). All colours are
// linear-space RGB and slightly oversaturated: the renderer tone-maps with
// ACESFilmicToneMapping under warm key/rim lights, which desaturates and
// shifts hues toward gold, so the base colours lean past the target to land
// right on screen.

import * as THREE from "three";

// ── Shared slot specs (identical on all four robots) ───────────────────
// sRGB targets from the reference photo, converted to linear with a
// slight saturation boost (x1.12) so ACES lands on the clean tint.
const AMBER_YELLOW = { color: [1.0, 0.413, 0.007], roughness: 0.4, metalness: 0.0 };   // #ffb52e
const BRIGHT_ORANGE = { color: [1.0, 0.144, 0.008], roughness: 0.45, metalness: 0.0 }; // #ff7a2f
const AMBER = { color: [0.847, 0.339, 0.022], roughness: 0.45, metalness: 0.0 };       // #eda63e
const CLEAN_YELLOW = { color: [1.0, 0.608, 0.021], roughness: 0.4, metalness: 0.0 };   // #ffd23f
const CREAM = { color: [0.888, 0.86, 0.798], roughness: 0.35, metalness: 0.0 };        // #f2efe8
const DARK = { color: [0.012, 0.012, 0.014], roughness: 0.55, metalness: 0.3 };        // #1d1d1f
const GRAY = { color: [0.256, 0.256, 0.279], roughness: 0.5, metalness: 0.35 };        // #8b8b90
// Camera-lens eye: very dark blue-black, glossy like coated glass.
const LENS = { color: [0.01, 0.012, 0.02], roughness: 0.05, metalness: 0.0 };

// Per-variant colours.
const WARM_GRAY = { color: [0.328, 0.312, 0.283], roughness: 0.35, metalness: 0.0 };      // #9b9892
const AMBER_ORANGE = { color: [0.815, 0.321, 0.019], roughness: 0.5, metalness: 0.0 };    // #e9a23b
const CHARCOAL = { color: [0.028, 0.028, 0.033], roughness: 0.5, metalness: 0.0 };        // #2f2f33
const CHARCOAL_FACE = { color: [0.044, 0.044, 0.051], roughness: 0.5, metalness: 0.0 };   // #3c3c40
const CHARCOAL_BODY = { color: [0.017, 0.017, 0.019], roughness: 0.5, metalness: 0.0 };   // #232326
const LIGHT_WARM_GRAY = { color: [0.485, 0.459, 0.416], roughness: 0.35, metalness: 0.0 };// #b9b5ae
const PURPLE_RING = { color: [0.114, 0.058, 0.578], roughness: 0.4, metalness: 0.0 };     // #6a52c8
const PURPLE_FEET = { color: [0.164, 0.085, 0.584], roughness: 0.45, metalness: 0.0 };    // #7a5fc9
const PALE_BLUE = { color: [0.441, 0.613, 0.723], roughness: 0.35, metalness: 0.0 };      // #b6cfdd
const MEDIUM_BLUE = { color: [0.13, 0.321, 0.552], roughness: 0.35, metalness: 0.0 };     // #6f9ec4
const BLUE_GRAY = { color: [0.183, 0.379, 0.565], roughness: 0.35, metalness: 0.0 };      // #7fa9c6
const FEET_YELLOW = { color: [0.784, 0.455, 0.023], roughness: 0.5, metalness: 0.0 };     // #e5b93e

// Slots (validated against the four-robot reference photo): headDome
// (top shell), facePlate (around the eye), trim (band under the shell +
// upper beak base), beakUpper (mouth-roof plate), beakLower (jaw plates),
// eyeRing, lens, bodyShell (trunk shells), legShells (upper-leg shells),
// feet (foot blocks + ankle brackets), soles, mechDark, mechGray.
export const VARIANTS = {
  // Front-center robot: warm gray head, cream body, amber eye, orange
  // trim/upper beak over a lighter amber lower beak, amber-orange feet.
  classic: {
    headDome: WARM_GRAY,
    facePlate: WARM_GRAY,
    trim: BRIGHT_ORANGE,
    beakUpper: BRIGHT_ORANGE,
    beakLower: AMBER,
    eyeRing: AMBER_YELLOW,
    lens: LENS,
    bodyShell: CREAM,
    legShells: CREAM,
    feet: AMBER_ORANGE,
    soles: AMBER_ORANGE,
    mechDark: DARK,
    mechGray: GRAY,
  },
  // Front-left robot: charcoal shells, slightly lighter face plate,
  // amber eye, orange/amber beak, yellow soles peeking under black feet.
  charcoal: {
    headDome: CHARCOAL,
    facePlate: CHARCOAL_FACE,
    trim: BRIGHT_ORANGE,
    beakUpper: BRIGHT_ORANGE,
    beakLower: AMBER,
    eyeRing: AMBER_YELLOW,
    lens: LENS,
    bodyShell: CHARCOAL_BODY,
    legShells: CHARCOAL_BODY,
    feet: CHARCOAL_BODY,
    soles: CLEAN_YELLOW,
    mechDark: DARK,
    mechGray: GRAY,
  },
  // Back-center robot: light warm gray head, purple eye, YELLOW trim and
  // beak (both plates), cream body, purple feet.
  purple: {
    headDome: LIGHT_WARM_GRAY,
    facePlate: LIGHT_WARM_GRAY,
    trim: CLEAN_YELLOW,
    beakUpper: CLEAN_YELLOW,
    beakLower: CLEAN_YELLOW,
    eyeRing: PURPLE_RING,
    lens: LENS,
    bodyShell: CREAM,
    legShells: CREAM,
    feet: PURPLE_FEET,
    soles: PURPLE_FEET,
    mechDark: DARK,
    mechGray: GRAY,
  },
  // Right robot: pale blue dome over a medium blue face plate (two
  // different blues), amber eye, orange/amber beak, blue-gray body,
  // clean yellow feet.
  blue: {
    headDome: PALE_BLUE,
    facePlate: MEDIUM_BLUE,
    trim: BRIGHT_ORANGE,
    beakUpper: BRIGHT_ORANGE,
    beakLower: AMBER,
    eyeRing: AMBER_YELLOW,
    lens: LENS,
    bodyShell: BLUE_GRAY,
    legShells: BLUE_GRAY,
    feet: FEET_YELLOW,
    soles: FEET_YELLOW,
    mechDark: DARK,
    mechGray: GRAY,
  },
};

export const VARIANT_NAMES = Object.keys(VARIANTS);
export const DEFAULT_VARIANT = "classic";

export const randomVariantName = () =>
  VARIANT_NAMES[Math.floor(Math.random() * VARIANT_NAMES.length)];

// Mesh filename -> material spec for one variant (mjlab model meshes).
// Anything not listed is small mechanical hardware and falls back to
// mechGray. Verified visually against the four-robot reference photo.
export function meshMaterialsFor(v) {
  return {
    // Head: dome, band under it, face plate around the eye.
    "top_head_shell.stl": v.headDome,
    "bottom_head_shell.stl": v.trim,
    "face_part.stl": v.facePlate,
    // The eye: printed ring + camera lens behind it.
    "noenoeil.stl": v.eyeRing,
    "lens.stl": v.lens,
    "m12_lens_holder.stl": v.mechDark,
    // Beak: mouth-roof plate on top, rigid jaw + soft pad below.
    "soft_mouth_top.stl": v.beakUpper,
    "jaw.stl": v.beakLower,
    "jaw_soft.stl": v.beakLower,
    // Body shells
    "trunk_base.stl": v.bodyShell,
    "left_shell.stl": v.bodyShell,
    "right_shell.stl": v.bodyShell,
    "upper_leg_left.stl": v.legShells,
    "upper_leg_right.stl": v.legShells,
    "hip_l.stl": v.trim,
    // Feet: foot block + ankle bracket, soft sole pad below.
    "foot_left.stl": v.feet,
    "foot_right.stl": v.feet,
    "ankle_left.stl": v.feet,
    "ankle_right.stl": v.feet,
    "sole_left.stl": v.soles,
    "sole_right.stl": v.soles,
    // Dark mechanics
    "xl330.stl": v.mechDark,
    "neck.stl": v.mechDark,
    "np_f970.stl": v.mechDark,
    "pcb__raspberry_pi_zero_2_w.stl": v.mechDark,
    "elec_rpi_robot_hat_pcb.stl": v.mechDark,
    "banana_pcb_locker.stl": v.mechDark,
    "speaker.stl": v.mechDark,
    // Gray mechanics
    "upper_leg_rigidity_plate.stl": v.mechGray,
    "leg.stl": v.mechGray,
    "yaw2roll.stl": v.mechGray,
    "yaw_roll_motion.stl": v.mechGray,
    "neck_pitch.stl": v.mechGray,
    "bearing_roll.stl": v.mechGray,
    "motor_support.stl": v.mechGray,
    "power_support.stl": v.mechGray,
    "seeed_bearing__configuration__22x16x4.stl": v.mechGray,
    "seeed_bearing__configuration_default.stl": v.mechGray,
  };
}

// buildRig materialForMesh hook for one variant.
export const materialHookFor = (v) => {
  const map = meshMaterialsFor(v);
  return (mesh) => map[mesh] ?? v.mechGray;
};

// ── Live re-skin ────────────────────────────────────────────────────────
// Swap materials on an already-built rig without reloading any STL.
// Meshes are identified via userData.meshName (set by duck.js buildRig and
// preserved by cloneRig). Materials are cached per resolved spec so clones
// and repeated switches share GPU material instances.
const matCache = new Map();
function matFor(spec) {
  const key = `${spec.color.join(",")}|${spec.roughness ?? 0.5}|${spec.metalness ?? 0}`;
  let m = matCache.get(key);
  if (!m) {
    m = new THREE.MeshStandardMaterial({
      color: new THREE.Color(...spec.color),
      roughness: spec.roughness ?? 0.5,
      metalness: spec.metalness ?? 0.0,
    });
    matCache.set(key, m);
  }
  return m;
}

export function applyVariant(rig, variant) {
  const v = typeof variant === "string" ? VARIANTS[variant] : variant;
  const map = meshMaterialsFor(v);
  rig.root.traverse((o) => {
    if (!o.isMesh || !o.userData.meshName) return;
    o.material = matFor(map[o.userData.meshName] ?? v.mechGray);
  });
}

// Linear-space spec colour -> sRGB CSS hex, for swatch UI elements.
export function specToHex(spec) {
  return `#${new THREE.Color(...spec.color).getHexString()}`;
}
