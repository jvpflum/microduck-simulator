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
| Roulade | `roulade.onnx` | Rolls over and recovers |

Policies and MJCF model from
[apirrone/microduck_runtime](https://github.com/apirrone/microduck_runtime)
and [apirrone/mjlab_microduck](https://github.com/apirrone/mjlab_microduck).

## Controls

- Arrow up / down: forward / back
- Arrow left / right: turn
- A / E: strafe
- R: reset
- Drag to orbit, scroll to zoom
- Colour dots: repaint the duck (it quacks)

## How it works

- `rl.js` fetches the MJCF (`robot_allcollisions.xml`), strips the visual
  geoms, injects a floor and a STAND keyframe, and compiles it with the
  official `@mujoco/mujoco` WASM bindings.
- The 61D observation (gyro, projected gravity, joint pos/vel, last action,
  command) matches `mjlab_microduck/scripts/infer_policy.py`.
- Rendering is a three.js rig built from `kinematics.json` + decimated STL
  meshes, driven directly from MuJoCo `qpos`.
