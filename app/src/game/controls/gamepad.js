// Gamepad input source (see controller.js for the source interface
// contract). Same mapping as the robot runtime (microduck_runtime):
//
//   Left stick   vertical = vx (asymmetric fwd/back), horizontal = turn,
//                EMA-smoothed like the runtime's cmd_alpha.
//   Right stick  camera orbit rate (reported raw in axes.orbitX/Y; the
//                velocity smoothing / coasting lives downstream in the
//                camera code). R3 toggles the chase cam.
//   X            roll (crouch-glide in roller mode - game decides)
//   Y            ball spawn/respawn
//   RB / LB      right / left kick
//   DpadDown     sit <-> stand
//   DpadUp       short press = back to run; HOLD ~1 s = legs <-> rollers
//                (like the robot's 3 s hold, shortened for the web)
//   RT / LT      analog jaw (max of both); RT edge quacks through a
//                Schmitt trigger (fire at >= 0.35, re-arm below 0.2 - a
//                single threshold re-fires on jitter, which is how one
//                squeeze used to quack several times).

const PAD_DEADZONE = 0.15;
const PAD_ALPHA = 0.12; // EMA smoothing toward the stick target
const dz = (v) => (Math.abs(v) < PAD_DEADZONE ? 0 : v);

// Standard-mapping button indices.
const BTN_X = 2, BTN_Y = 3, BTN_LB = 4, BTN_RB = 5, BTN_LT = 6, BTN_RT = 7;
const BTN_R3 = 11, BTN_DPAD_UP = 12, BTN_DPAD_DOWN = 13;

const DPAD_UP_HOLD_MS = 1000; // hold-to-switch-loco duration

export class GamepadSource {
  id = "gamepad";
  connected = false;
  command = new Float32Array(3); // [vx, 0, wz], EMA-smoothed
  axes = { jaw: 0, orbitX: 0, orbitY: 0 };
  pressed = {
    x: false, y: false, rb: false, lb: false, r3: false,
    dpadDown: false, dpadUp: false,
  };
  onAction = () => {}; // assigned by the Controller at registration

  #getVelocityLimits;
  #active = false; // owns twist authority (stick input, until EMA settles)
  #dpadUpAt = 0; // wall-clock of the current DpadUp press
  #dpadUpFired = false; // latch: one loco switch per hold
  #rtArmed = true; // Schmitt trigger state for the RT quack

  constructor({ getVelocityLimits }) {
    this.#getVelocityLimits = getVelocityLimits;
  }

  init() {} // poll-based: nothing to attach
  dispose() {}

  isActive() {
    return this.#active;
  }

  poll() {
    const prev = this.pressed;
    const gp = [...(navigator.getGamepads?.() ?? [])].find((p) => p && p.connected);
    this.connected = !!gp;
    if (!gp) {
      if (this.#active) {
        this.#active = false;
        this.command.fill(0);
        this.axes.jaw = 0;
      }
      this.axes.orbitX = 0;
      this.axes.orbitY = 0;
      return;
    }
    const now = performance.now();

    // Left stick only: vertical = forward/back, horizontal = turn.
    // (No strafe; the right stick doesn't drive movement.)
    const lx = dz(gp.axes[0] ?? 0), ly = dz(gp.axes[1] ?? 0);
    const up = -ly; // browser sticks report up as -1
    const [limF, limB, limA] = this.#getVelocityLimits();
    const target = [
      up >= 0 ? up * limF : up * -limB,
      0,
      -lx * limA,
    ];
    for (let i = 0; i < 3; i++) this.command[i] += PAD_ALPHA * (target[i] - this.command[i]);
    // Sticks grab command authority on first input, release when back at
    // rest (then the keyboard takes over again through the Controller's
    // arbitration).
    const stickInput = lx !== 0 || ly !== 0;
    if (stickInput) this.#active = true;
    else if (
      this.#active &&
      Math.abs(this.command[0]) + Math.abs(this.command[1]) + Math.abs(this.command[2]) < 0.01
    ) {
      this.#active = false;
      this.command.fill(0);
    }

    // Right stick: raw (deadzoned) orbit rate. Reported every frame - the
    // downstream camera step needs the zeros too so a released stick
    // coasts to a stop.
    this.axes.orbitX = dz(gp.axes[2] ?? 0);
    this.axes.orbitY = dz(gp.axes[3] ?? 0);

    // R3 (right stick click): chase-cam toggle, gamepad twin of KeyC.
    const r3 = !!gp.buttons[BTN_R3]?.pressed;
    if (r3 && !prev.r3) this.onAction("chaseToggle");
    prev.r3 = r3;

    const x = !!gp.buttons[BTN_X]?.pressed;
    if (x && !prev.x) this.onAction("roll");
    prev.x = x;

    const y = !!gp.buttons[BTN_Y]?.pressed;
    if (y && !prev.y) this.onAction("spawnBall");
    prev.y = y;

    const rb = !!gp.buttons[BTN_RB]?.pressed;
    if (rb && !prev.rb) this.onAction("kickR");
    prev.rb = rb;
    const lb = !!gp.buttons[BTN_LB]?.pressed;
    if (lb && !prev.lb) this.onAction("kickL");
    prev.lb = lb;

    const dpadDown = !!gp.buttons[BTN_DPAD_DOWN]?.pressed;
    if (dpadDown && !prev.dpadDown) this.onAction("sitToggle");
    prev.dpadDown = dpadDown;

    // DpadUp: short press = back to run; hold fires ONE loco switch.
    const dpadUp = !!gp.buttons[BTN_DPAD_UP]?.pressed;
    if (dpadUp && !prev.dpadUp) {
      this.#dpadUpAt = now;
      this.#dpadUpFired = false;
      this.onAction("walk");
    }
    if (dpadUp && !this.#dpadUpFired && now - this.#dpadUpAt >= DPAD_UP_HOLD_MS) {
      this.#dpadUpFired = true; // latch: one switch per hold
      this.onAction("locoToggle");
    }
    prev.dpadUp = dpadUp;

    // Triggers drive the mouth (max of both); RT quacks on its
    // Schmitt-triggered rising edge.
    const rt = gp.buttons[BTN_RT]?.value ?? 0;
    const lt = gp.buttons[BTN_LT]?.value ?? 0;
    this.axes.jaw = Math.max(rt, lt);
    if (this.#rtArmed && rt >= 0.35) {
      this.onAction("quack");
      this.#rtArmed = false;
    } else if (!this.#rtArmed && rt < 0.2) {
      this.#rtArmed = true;
    }
  }
}
