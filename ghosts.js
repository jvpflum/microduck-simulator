// Live multiplayer ghosts: every open tab of the sandbox broadcasts its
// duck state over WebRTC and renders the other visitors as translucent
// ducks. There is no backend - this is a static Space - so peer discovery
// uses Trystero's serverless signaling over public Nostr relays. Payloads
// are tiny (22 floats + a variant name at 10 Hz), and at most MAX_GHOSTS
// ghosts are instantiated regardless of how many peers are in the room.
//
// This module deliberately imports nothing from duck.js / variants.js:
// on the private Space those need the ?__sign JWT that only rl.js knows
// how to append, so all rig helpers arrive through init parameters.

const TRYSTERO_URL = "https://esm.run/trystero@0.25.3/nostr";
const APP_ID = "microduck-sandbox";
const ROOM = "lobby";
const MAX_GHOSTS = 3;
const SEND_HZ = 10;
const GHOST_OPACITY = 0.35;
// Per-frame easing toward the last received state (10 Hz updates, ~60 fps
// rendering): position/joints converge in ~4 frames, fast but smooth.
const EASE = 0.25;

export async function initGhosts(env) {
  const noop = { update() {}, peerCount: () => 0, ghostCount: () => 0, debug: () => [] };
  let room;
  try {
    const { joinRoom } = await import(TRYSTERO_URL);
    room = joinRoom({ appId: APP_ID }, ROOM);
  } catch (e) {
    console.warn("ghosts disabled (signaling unavailable):", e);
    return noop;
  }

  const { scene, cloneRig, setJoint, setJawOpen, applyVariant, jointNames, getLocalState } = env;
  // trystero 0.25 (backed by @trystero-p2p): makeAction returns
  // { send, onMessage, onReceiveProgress } where onMessage is a SETTER -
  // the receive handler is registered by assignment, not by calling it.
  const act = room.makeAction("s");
  // send() rejects when a peer's channel drops mid-transfer; that peer will
  // be swept anyway, so failures are non-events.
  const sendState = (data) => act.send(data).catch(() => {});

  // Ghosts share geometry with the local rig (cloneRig), but get their own
  // transparent material clones so repainting them never touches the
  // visitor's own duck (variants.js caches materials globally).
  const ghostify = (rig) => {
    const cache = new Map();
    rig.root.traverse((o) => {
      if (!o.isMesh) return;
      let m = cache.get(o.material.uuid);
      if (!m) {
        m = o.material.clone();
        m.transparent = true;
        m.opacity = GHOST_OPACITY;
        m.depthWrite = false;
        cache.set(o.material.uuid, m);
      }
      o.material = m;
    });
  };

  const ghosts = new Map(); // peerId -> ghost
  const makeGhost = (state) => {
    const rig = cloneRig(env.rig);
    applyVariant(rig, state.v);
    ghostify(rig);
    scene.add(rig.placer);
    const trunk = rig.bodies.get("trunk_base");
    // Snap straight to the first received pose: no fly-in from the origin.
    trunk.position.set(state.p[0], state.p[1], state.p[2]);
    trunk.quaternion.set(state.p[4], state.p[5], state.p[6], state.p[3]);
    return { rig, trunk, target: state, joints: state.j.slice(), jaw: state.w ?? 0, variant: state.v };
  };

  const removeGhost = (peerId) => {
    const g = ghosts.get(peerId);
    if (!g) return;
    scene.remove(g.rig.placer);
    const seen = new Set();
    g.rig.root.traverse((o) => {
      if (o.isMesh && !seen.has(o.material.uuid)) {
        seen.add(o.material.uuid);
        o.material.dispose();
      }
    });
    ghosts.delete(peerId);
  };

  // Handler signature in this build: (payload, context) where context is
  // { peerId } - NOT the bare peerId string of classic trystero.
  act.onMessage = (state, { peerId }) => {
    let g = ghosts.get(peerId);
    if (!g) {
      if (ghosts.size >= MAX_GHOSTS) return; // room stays open, rendering capped
      // Nostr relays replay recent events, so states from already-dead
      // sessions can arrive right after joining: only ghost live peers.
      if (!(peerId in room.getPeers())) return;
      g = makeGhost(state);
      ghosts.set(peerId, g);
      g.lastSeen = performance.now();
      return;
    }
    g.lastSeen = performance.now();
    g.target = state;
    if (state.v !== g.variant) {
      g.variant = state.v;
      applyVariant(g.rig, state.v);
      ghostify(g.rig);
    }
  };

  // Same setter-style registration as onMessage.
  room.onPeerLeave = (peerId) => removeGhost(peerId);

  // Graceful exit so other tabs drop this ghost immediately...
  window.addEventListener("pagehide", () => {
    try { room.leave(); } catch {}
  });
  // ...and a staleness sweep for peers that vanished without leaving
  // (crashed tab, dropped connection): 5 s without a state packet at
  // 10 Hz means the peer is gone, not just lagging.
  const STALE_MS = 5000;

  // Broadcast + stale sweep on setInterval (not rAF) so both keep running
  // in occluded tabs.
  setInterval(() => {
    const s = getLocalState();
    if (s) sendState(s);
    const now = performance.now();
    for (const [peerId, g] of [...ghosts]) {
      if (now - g.lastSeen > STALE_MS) removeGhost(peerId);
    }
  }, 1000 / SEND_HZ);

  // Scratch THREE.Quaternion without importing three: slerp needs a real
  // instance, cloned here from any node of the already-built local rig.
  const _q = env.rig.placer.quaternion.clone();
  return {
    room,
    peerCount: () => Object.keys(room.getPeers()).length,
    ghostCount: () => ghosts.size,
    debug: () => [...ghosts.values()].map((g) => {
      let meshes = 0, visible = 0, op = null;
      g.rig.root.traverse((o) => {
        if (o.isMesh) { meshes++; if (o.visible) visible++; op ??= o.material.opacity; }
      });
      const w = g.trunk.getWorldPosition(g.trunk.position.clone());
      return { p: g.trunk.position.toArray(), world: w.toArray(), inScene: !!g.rig.placer.parent, meshes, visible, op, v: g.variant };
    }),
    update() {
      for (const g of ghosts.values()) {
        const t = g.target;
        // Trunk pose in raw MJCF coords: the cloned root applies Z-up -> Y-up.
        g.trunk.position.x += (t.p[0] - g.trunk.position.x) * EASE;
        g.trunk.position.y += (t.p[1] - g.trunk.position.y) * EASE;
        g.trunk.position.z += (t.p[2] - g.trunk.position.z) * EASE;
        _q.set(t.p[4], t.p[5], t.p[6], t.p[3]); // THREE order x,y,z,w
        g.trunk.quaternion.slerp(_q, EASE);
        for (let i = 0; i < jointNames.length; i++) {
          g.joints[i] += (t.j[i] - g.joints[i]) * EASE;
          setJoint(g.rig, jointNames[i], g.joints[i]);
        }
        g.jaw += ((t.w ?? 0) - g.jaw) * EASE;
        setJawOpen(g.rig, g.jaw);
      }
    },
  };
}
