---
title: Microduck Sandbox
emoji: 🐤
colorFrom: yellow
colorTo: gray
sdk: static
pinned: false
---

# Microduck RL playground

Real trained RL policies for the Microduck robot, running fully in the
browser: MuJoCo compiled to WebAssembly steps the physics, onnxruntime-web
runs the policy network at 50 Hz. No server, no backend.

Two locomotion variants of the same robot are included: **legs** (walking,
the default) and **rollers** (the wheeled skating variant). Press `M` (or
hold D-pad up ~1 s on a gamepad) to switch; the roller model, meshes and
policies are lazy-loaded on the first switch.

## Policies

| Mode | Checkpoint | What it does |
|--------|-----------|--------------|
| Run (legs) | `BEST_alpha_walking.onnx` | Velocity-tracking locomotion (arrows / WASD to steer) |
| Sit | `BEST_alpha_sitstand.onnx` | Sits down on its hull, stands back up |
| Roll | `roulade.onnx` | Rolls over and recovers |
| Kick | `ball_kick_left.onnx` / `ball_kick_right.onnx` | Blind one-shot kick (0.5 s window, zeroed commands), left or right leg |
| Drive (rollers) | `BEST_roller.onnx` | Velocity-tracking skating on 4 passive wheels (higher top speed: 0.6 m/s) |
| Crouch (rollers) | `BEST_roller_crouch.onnx` | One-shot crouch-glide: sinks low over ~3.5 s and stands back up (phase-encoded command) |

Policies and MJCF model from
[apirrone/microduck_runtime](https://github.com/apirrone/microduck_runtime)
and [apirrone/mjlab_microduck](https://github.com/apirrone/mjlab_microduck).

## Controls

- Arrows or WASD (ZQSD): forward / back + turn
- M: switch legs <-> rollers
- Q / E (A / E on AZERTY): kick left / right (legs only)
- R: roll (legs) / crouch-glide (rollers)
- B: pop / respawn a kickable ball in front of the duck
- C: toggle the chase camera (on by default; dragging detaches it)
- Space: reset
- Drag to orbit, scroll to zoom
- Colour dots: repaint the duck (it quacks)

In roller mode the legs-only actions (kicks, sit) are disabled and their
hints fade out; play the ball by driving into it.

The ball is local-only: it lives in your tab's physics and is not shared
with the multiplayer ghosts. A square arena (3 x 3 m) fences the play
area so neither the ball nor the duck can wander off.

### Gamepad

Plug in a controller and the same mapping as the real robot runtime applies:

- Left stick: forward / back + turn
- Right stick: orbit the camera (detaches the chase cam)
- R3 (right stick click): toggle the chase cam back on
- X: roll (legs) / crouch-glide (rollers)
- Y: pop / respawn the ball
- RB / LB: right / left kick (legs only)
- D-pad down: sit / stand toggle (legs only)
- D-pad up: hold ~1 s to switch legs <-> rollers (the real robot uses a 3 s hold)
- Right trigger: mouth (analog) + quack

## Multiplayer ghosts

Other people visiting the Space at the same time show up as translucent
ducks, live. Peer-to-peer WebRTC via [Trystero](https://github.com/dmotz/trystero)
(serverless signaling over public Nostr relays), so it works from a static
Space with no backend. Each tab broadcasts its duck's pose (trunk + 14
joints + jaw + colour + locomotion variant) at 10 Hz; up to 3 ghosts are
rendered, extra peers stay connected but invisible. The camcorder-style OSD
in the top-right corner shows "N ONLINE" when peers are around.

Ghost limitations in v1: a peer in roller mode renders with the roller rig
only if your tab has already loaded it (otherwise it falls back to the leg
rig), and ghost wheels don't spin (passive wheel joints aren't broadcast).
Old clients simply ignore the new variant flag.

## How it works

- `rl.js` fetches the MJCF (`robot_allcollisions.xml`, or
  `robot_allcollisions_rollers.xml` for the roller variant), strips the
  visual geoms, injects a floor, arena walls, a ball and a STAND keyframe,
  and compiles it with the official `@mujoco/mujoco` WASM bindings.
- Both variants share the exact same policy interface: 61D observation
  (gyro, projected gravity, 14 joint pos/vel, last action, 13D command)
  and 14 position-targets, matching `mjlab_microduck/scripts/infer_policy.py`.
  The roller variant adds 4 passive wheel hinges that appear in `qpos`
  (zeroed in the keyframe) but not in the observation.
- Rendering is a three.js rig built from `kinematics.json` /
  `kinematics_rollers.json` + decimated STL meshes, driven directly from
  MuJoCo `qpos` (including the passive wheel spin).
- Switching variants swaps the compiled model + data + rig + ONNX sessions
  wholesale; both stay resident after the first load so toggling back and
  forth is instant.
