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

## Policies

| Button | Checkpoint | What it does |
|--------|-----------|--------------|
| Run | `BEST_alpha_walking.onnx` | Velocity-tracking locomotion (arrows / WASD to steer) |
| Sit | `BEST_alpha_sitstand.onnx` | Sits down on its hull, stands back up |
| Roll | `roulade.onnx` | Rolls over and recovers |
| Kick | `ball_kick_left.onnx` / `ball_kick_right.onnx` | Blind one-shot kick (0.5 s window, zeroed commands), left or right leg |

Policies and MJCF model from
[apirrone/microduck_runtime](https://github.com/apirrone/microduck_runtime)
and [apirrone/mjlab_microduck](https://github.com/apirrone/mjlab_microduck).

## Controls

- Arrows or WASD (ZQSD): forward / back + turn
- Q / E (A / E on AZERTY): kick left / right
- R: roll (one barrel roll, then back to running)
- B: pop / respawn a kickable ball in front of the duck
- C: toggle the chase camera (on by default; dragging detaches it)
- Space: reset
- Drag to orbit, scroll to zoom
- Colour dots: repaint the duck (it quacks)

The ball is local-only: it lives in your tab's physics and is not shared
with the multiplayer ghosts. A square arena (3 x 3 m) fences the play
area so neither the ball nor the duck can wander off.

### Gamepad

Plug in a controller and the same mapping as the real robot runtime applies:

- Left stick: forward / back + turn
- Right stick: orbit the camera (detaches the chase cam)
- R3 (right stick click): toggle the chase cam back on
- X: roll
- Y: pop / respawn the ball
- RB / LB: right / left kick
- D-pad down: sit / stand toggle
- Right trigger: mouth (analog) + quack

## Multiplayer ghosts

Other people visiting the Space at the same time show up as translucent
ducks, live. Peer-to-peer WebRTC via [Trystero](https://github.com/dmotz/trystero)
(serverless signaling over public Nostr relays), so it works from a static
Space with no backend. Each tab broadcasts its duck's pose (trunk + 14
joints + jaw + colour) at 10 Hz; up to 3 ghosts are rendered, extra peers
stay connected but invisible. The header shows "N online" when peers are
around.

## How it works

- `rl.js` fetches the MJCF (`robot_allcollisions.xml`), strips the visual
  geoms, injects a floor and a STAND keyframe, and compiles it with the
  official `@mujoco/mujoco` WASM bindings.
- The 61D observation (gyro, projected gravity, joint pos/vel, last action,
  command) matches `mjlab_microduck/scripts/infer_policy.py`.
- Rendering is a three.js rig built from `kinematics.json` + decimated STL
  meshes, driven directly from MuJoCo `qpos`.
