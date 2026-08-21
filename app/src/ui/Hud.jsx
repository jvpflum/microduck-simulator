// In-game HUD: Back (top-left), mode readout (top-right, Matrix-style
// letter scramble on change), quickbar (bottom-left: live colour dots +
// legs/rollers segment), telemetry stack (bottom-right) and the
// LOADING ROLLERS line while the roller stack streams in.
//
// Whole HUD is hidden while the title/pause overlay is up; touch mode
// strips it down to the thumbs + Back.
import { useEffect, useRef, useState } from "react";
import Box from "@mui/material/Box";
import ButtonBase from "@mui/material/ButtonBase";
import { keyframes } from "@mui/material/styles";
import { useGame, gameApi } from "../store.js";
import { VARIANT_SWATCHES } from "../game/game.js";
import { INK, ORANGE, MONO } from "../theme.js";

// Matrix-style letter scramble: on change every glyph flips through random
// charset entries, then locks to its target left-to-right over ~0.45 s.
// Monospace keeps the width stable mid-scramble.
const SCRAMBLE_GLYPHS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789#$%&*<>/=+";
function useScramble(target) {
  const [text, setText] = useState(target.toUpperCase());
  const timer = useRef(null);
  const shown = useRef(target.toUpperCase());
  useEffect(() => {
    const t = target.toUpperCase();
    if (shown.current === t) return;
    shown.current = t;
    if (timer.current) clearInterval(timer.current);
    const n = t.length;
    const DUR = 450; // ms until the last letter locks
    const rnd = () => SCRAMBLE_GLYPHS[(Math.random() * SCRAMBLE_GLYPHS.length) | 0];
    const t0 = performance.now();
    timer.current = setInterval(() => {
      // Letters 0..k-1 are locked; the rest keep boiling.
      const k = Math.floor(((performance.now() - t0) / DUR) * n);
      if (k >= n) {
        clearInterval(timer.current);
        timer.current = null;
        setText(t);
        return;
      }
      let out = t.slice(0, k);
      for (let i = k; i < n; i++) out += rnd();
      setText(out);
    }, 40);
    return () => timer.current && clearInterval(timer.current);
  }, [target]);
  return text;
}

const recBlink = keyframes`
  0%, 58% { opacity: 1; }
  59%, 100% { opacity: 0.12; }
`;

function BackButton() {
  return (
    <ButtonBase
      onClick={() => useGame.setState({ menuOpen: true })}
      sx={{
        position: "fixed",
        top: "1.4rem",
        left: "1.6rem",
        zIndex: 10,
        display: "inline-flex",
        alignItems: "center",
        gap: "0.45em",
        border: "1px solid rgba(255, 255, 255, 0.16)",
        background: "rgba(8, 8, 12, 0.55)",
        color: "rgba(255, 255, 255, 0.75)",
        font: "inherit",
        fontSize: "0.8rem",
        fontWeight: 600,
        letterSpacing: "0.14em",
        textTransform: "uppercase",
        p: "0.6rem 1.15rem",
        borderRadius: "999px",
        transition: "color 0.15s ease, background 0.15s ease, border-color 0.15s ease",
        "&:hover": {
          color: "#fff",
          borderColor: "rgba(255, 255, 255, 0.4)",
          background: "rgba(255, 255, 255, 0.06)",
        },
        "&.Mui-focusVisible": {
          outline: `2px solid ${ORANGE}`,
          outlineOffset: 2,
        },
      }}
    >
      {"\u2190"} Back
    </ButtonBase>
  );
}

// Active policy/mode readout, top-right: bright orange caption over a big
// white monospace name.
function ModeReadout() {
  const modeLabel = useGame((s) => s.modeLabel);
  const text = useScramble(modeLabel);
  return (
    <Box
      sx={{
        position: "fixed",
        top: "1.4rem",
        right: "1.6rem",
        zIndex: 10,
        fontFamily: MONO,
        pointerEvents: "none",
        textAlign: "right",
      }}
    >
      <Box
        component="span"
        sx={{
          display: "block",
          mb: "0.35rem",
          fontSize: "0.78rem",
          fontWeight: 600,
          letterSpacing: "0.24em",
          textTransform: "uppercase",
          color: ORANGE,
          textShadow: "0 0 10px rgba(255, 122, 47, 0.35)",
        }}
      >
        Mode
      </Box>
      <Box
        sx={{
          minWidth: "8ch",
          minHeight: "1.2em",
          fontSize: "1.7rem",
          fontWeight: 600,
          letterSpacing: "0.16em",
          textTransform: "uppercase",
          color: "rgba(255, 255, 255, 0.95)",
          textAlign: "right",
          textShadow: "0 0 12px rgba(255, 255, 255, 0.28)",
          lineHeight: 1.2,
        }}
      >
        {text}
      </Box>
    </Box>
  );
}

// In-game quick picks, bottom-left: colour dots + legs/rollers, live.
// Same pill language as the vitrine DuckPlayground.
function Quickbar() {
  const variant = useGame((s) => s.variant);
  const locoWant = useGame((s) => s.locoWant);
  const pill = {
    display: "inline-flex",
    gap: "0.24rem",
    p: "0.26rem",
    border: "1px solid rgba(255, 255, 255, 0.14)",
    borderRadius: "999px",
    background: "rgba(8, 8, 12, 0.55)",
  };
  return (
    <Box
      sx={{
        position: "fixed",
        bottom: "1.4rem",
        left: "1.6rem",
        zIndex: 10,
        display: "flex",
        flexWrap: "wrap",
        alignItems: "center",
        gap: "0.6rem",
      }}
    >
      <Box role="group" aria-label="Duck colours" sx={pill}>
        {Object.entries(VARIANT_SWATCHES).map(([name, hex]) => (
          <ButtonBase
            key={name}
            aria-label={`${name} colours`}
            aria-pressed={name === variant}
            onClick={() => gameApi.setVariant?.(name)}
            sx={{
              height: { xs: "2rem", sm: "2.3rem" },
              px: { xs: "0.42rem", sm: "0.5rem" },
              borderRadius: "999px",
              "&.Mui-focusVisible": { outline: `2px solid ${ORANGE}`, outlineOffset: 2 },
              "&:hover .dot": { opacity: 1 },
            }}
          >
            <Box
              className="dot"
              sx={{
                width: { xs: "1.2rem", sm: "1.4rem" },
                height: { xs: "1.2rem", sm: "1.4rem" },
                borderRadius: "50%",
                background: hex,
                opacity: name === variant ? 1 : 0.45,
                transform: name === variant ? "scale(1.15)" : "none",
                transition: "transform 0.2s ease, opacity 0.2s ease",
              }}
            />
          </ButtonBase>
        ))}
      </Box>
      <Box role="group" aria-label="Locomotion mode" sx={pill}>
        {["legs", "rollers"].map((name) => (
          <ButtonBase
            key={name}
            aria-pressed={locoWant === name}
            onClick={() => gameApi.requestLoco?.(name)}
            sx={{
              height: { xs: "2rem", sm: "2.3rem" },
              px: { xs: "0.8rem", sm: "1.05rem" },
              borderRadius: "999px",
              font: "inherit",
              fontSize: { xs: "0.68rem", sm: "0.76rem" },
              fontWeight: 600,
              letterSpacing: "0.14em",
              textTransform: "uppercase",
              color: locoWant === name ? ORANGE : "rgba(255, 255, 255, 0.45)",
              background: locoWant === name ? "rgba(255, 122, 47, 0.14)" : "transparent",
              transition: "color 0.2s ease, background 0.2s ease",
              "&.Mui-focusVisible": { outline: `2px solid ${ORANGE}`, outlineOffset: 2 },
            }}
          >
            {name === "legs" ? "Legs" : "Rollers"}
          </ButtonBase>
        ))}
      </Box>
    </Box>
  );
}

// Telemetry stack, bottom-right: peers / speed + odometer / loop rates,
// one line each, quietest at the bottom.
function Telemetry() {
  const t = useGame((s) => s.telemetry);
  const odo = t.odo < 1000 ? `${t.odo.toFixed(1)}M` : `${(t.odo / 1000).toFixed(2)}KM`;
  const lines = [];
  if (t.peers) lines.push(`${t.peers + 1} ONLINE`);
  lines.push(`${t.speed.toFixed(2)}M/S \u00b7 ODO ${odo}`);
  lines.push(`FPS ${t.fps} \u00b7 CTRL ${t.ctrlHz}HZ`);
  return (
    <Box
      sx={{
        position: "fixed",
        bottom: "1.4rem",
        right: "1.6rem",
        zIndex: 10,
        pointerEvents: "none",
        fontFamily: MONO,
        fontSize: "0.62rem",
        letterSpacing: "0.14em",
        textTransform: "uppercase",
        color: "rgba(255, 255, 255, 0.35)",
        fontVariantNumeric: "tabular-nums",
        textShadow: "0 0 6px rgba(255, 255, 255, 0.15)",
        whiteSpace: "pre-line",
        textAlign: "right",
        lineHeight: 1.8,
      }}
    >
      {lines.join("\n")}
    </Box>
  );
}

// BIOS-style line while the roller variant streams in on first switch.
function OsdLoad() {
  const rollersLoading = useGame((s) => s.rollersLoading);
  if (!rollersLoading) return null;
  return (
    <Box
      sx={{
        position: "fixed",
        top: "6.1rem", // below the mode readout
        right: "1.6rem",
        zIndex: 10,
        fontFamily: MONO,
        fontSize: "0.68rem",
        letterSpacing: "0.14em",
        color: ORANGE,
        textShadow: "0 0 8px rgba(255, 122, 47, 0.4)",
      }}
    >
      LOADING ROLLERS
      <Box
        component="span"
        sx={{ display: "inline-block", ml: "0.15em", animation: `${recBlink} 0.8s steps(1) infinite` }}
      >
        {"\u2588"}
      </Box>
    </Box>
  );
}

export default function Hud() {
  const entered = useGame((s) => s.entered);
  const menuOpen = useGame((s) => s.menuOpen);
  const touchMode = useGame((s) => s.touchMode);
  // HUD stays off until the first enter and while the title overlay is up
  // (the BIOS readout simply covers it during boot, same as before).
  if (!entered || menuOpen) return null;
  return (
    <>
      <BackButton />
      {/* Touch mode strips the HUD down to the thumbs: no quickbar, no
          mode readout, no telemetry. Back stays to reach the menu. */}
      {!touchMode && (
        <>
          <ModeReadout />
          <Quickbar />
          <Telemetry />
          <OsdLoad />
        </>
      )}
    </>
  );
}
