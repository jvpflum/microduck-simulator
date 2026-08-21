// Single-page title / pause menu: vitrine brand lockup, pitch, controls
// tutorial (keyboard / gamepad / touch variant) and the CTA. The first
// "Waddle in" cues the BIOS; later Esc / the in-game Back button reopen it
// as pause. Rows fade up once, in reading order - the stagger replays on
// every open because the component remounts.
import { useEffect, useRef, useState } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import ButtonBase from "@mui/material/ButtonBase";
import { keyframes, styled } from "@mui/material/styles";
import { useGame } from "../store.js";
import { signed } from "../game/signed.js";
import { INK, ORANGE } from "../theme.js";

const rowIn = keyframes`
  from { transform: translateY(12px); opacity: 0; }
  to { transform: none; opacity: 1; }
`;
const brandIn = keyframes`
  from { transform: translateY(10px) scale(0.94); opacity: 0; }
  to { transform: none; opacity: 1; }
`;

// One soft ease, small travel, ~80 ms between rows - staged, not showy.
const row = (delay, name = rowIn) => ({
  animation: `${name} 0.55s cubic-bezier(0.22, 1, 0.36, 1) both`,
  animationDelay: `${delay}s`,
  "@media (prefers-reduced-motion: reduce)": { animation: "none" },
});

const Kbd = styled("kbd")(({ round }) => ({
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  minWidth: round ? "1.9rem" : "1.65rem",
  height: "1.65rem",
  padding: "0 0.45rem",
  font: "inherit",
  fontSize: "0.68rem",
  fontWeight: 600,
  color: "#fff",
  background: "#14141c",
  border: `2px solid ${INK}`,
  borderRadius: round ? "50%" : 8,
  boxShadow: "0 0 0 2px rgba(255, 255, 255, 0.82)",
}));

const TILES = {
  kb: [
    { caps: "cluster-arrows", name: "Move", hint: "arrows or ZQSD" },
    { caps: ["A", "E"], name: "Kick", hint: "left / right" },
    { caps: ["R"], name: "Roll", hint: "barrel roll" },
    { caps: ["B"], name: "Ball", hint: "pop a fresh one" },
    { caps: ["C"], name: "Camera", hint: "toggle chase" },
    { caps: ["Space"], name: "Reset", hint: "fresh start" },
  ],
  pad: [
    { caps: ["LS"], name: "Move", hint: "left stick" },
    { caps: ["LB", "RB"], name: "Kick", hint: "left / right" },
    { caps: ["X"], name: "Roll", hint: "barrel roll" },
    { caps: ["Y"], name: "Ball", hint: "pop a fresh one" },
    { caps: ["RS"], name: "Camera", hint: "orbit" },
    { caps: ["\u2193"], name: "Sit", hint: "stand up" },
  ],
  touch: [
    { caps: ["Stick"], name: "Move", hint: "left thumb" },
    { caps: ["A"], round: true, name: "Kick", hint: "tap it" },
    { caps: ["B"], round: true, name: "Quack", hint: "hold for beak" },
  ],
};
const HINTS = {
  kb: "drag to orbit \u00b7 scroll to zoom",
  pad: "R3 chase \u00b7 RT quack",
  touch: "drag to orbit \u00b7 pinch to zoom",
};

function closeMenu() {
  useGame.setState({ menuOpen: false });
  if (!useGame.getState().entered) useGame.setState({ entered: true });
}

export default function TitleMenu() {
  const menuOpen = useGame((s) => s.menuOpen);
  const entered = useGame((s) => s.entered);
  const padConnected = useGame((s) => s.padConnected);
  const touchMode = useGame((s) => s.touchMode);
  const [closing, setClosing] = useState(false);
  const prevOpen = useRef(menuOpen);

  // Keep the overlay mounted through the 0.35 s closing fade.
  useEffect(() => {
    const was = prevOpen.current;
    prevOpen.current = menuOpen;
    if (was && !menuOpen) {
      setClosing(true);
      const t = setTimeout(() => setClosing(false), 380);
      return () => clearTimeout(t);
    }
  }, [menuOpen]);

  // Enter enters / resumes; Esc toggles the pause menu (never over the
  // BIOS readout).
  useEffect(() => {
    const onKey = (e) => {
      const s = useGame.getState();
      if (e.code === "Enter" && s.menuOpen) {
        if (e.target instanceof HTMLButtonElement && e.target.dataset.cta !== "1") return;
        e.preventDefault();
        closeMenu();
        return;
      }
      if (e.code !== "Escape") return;
      if (s.biosVisible) return;
      if (s.menuOpen) {
        if (s.entered) closeMenu();
        return;
      }
      if (s.entered) useGame.setState({ menuOpen: true });
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  if (!menuOpen && !closing) return null;

  // A plugged-in gamepad wins over the touch tutorial.
  const variant = padConnected ? "pad" : touchMode ? "touch" : "kb";
  const tiles = TILES[variant];

  return (
    <Box
      role="dialog"
      aria-modal="true"
      aria-label="Microduck"
      sx={{
        position: "fixed",
        inset: 0,
        zIndex: 30,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        p: "2.4rem 1.4rem 1.8rem",
        background: INK,
        opacity: menuOpen ? 1 : 0,
        pointerEvents: menuOpen ? "auto" : "none",
        overflowY: "auto",
        transition: "opacity 0.35s ease, background 0.4s ease",
      }}
    >
      <Box sx={{ width: "100%", maxWidth: "min(48rem, 92vw)", textAlign: "center" }}>
        {/* Vitrine brand lockup, centered above the title: drawn duck head
            + name. Hovering swaps to the open-beak frame, same hard sprite
            swap as the vitrine header. */}
        <Box
          aria-hidden
          sx={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            gap: "0.55rem",
            mb: "1.3rem",
            userSelect: "none",
            ...row(0, brandIn),
            "&:hover .duck-closed": { opacity: 0 },
            "&:hover .duck-open": { opacity: 1 },
          }}
        >
          <Box
            component="span"
            sx={{
              position: "relative",
              display: "block",
              height: { xs: "2.4rem", sm: "3rem" },
            }}
          >
            <Box
              component="img"
              className="duck-closed"
              alt=""
              src={signed("./assets/duck-head-mark.webp")}
              sx={{ display: "block", height: "100%", width: "auto" }}
            />
            {/* The open frame's canvas (460x333) is a hair wider/taller
                than the closed one (454x269): the dropped jaw sticks out
                past the head. Offsets pin the head in place while the jaw
                hangs below. */}
            <Box
              component="img"
              className="duck-open"
              alt=""
              src={signed("./assets/duck-head-mark-open.webp")}
              sx={{
                position: "absolute",
                top: "-0.75%",
                left: "-1.1%",
                width: "101.3%",
                maxWidth: "none",
                height: "auto",
                opacity: 0,
              }}
            />
          </Box>
          <Box
            component="span"
            sx={{
              fontWeight: 700,
              letterSpacing: "-0.03em",
              fontSize: { xs: "1.3rem", sm: "1.6rem" },
              color: "#fff",
            }}
          >
            MicroDuck
          </Box>
        </Box>

        <Box
          sx={{
            display: "flex",
            flexWrap: "wrap",
            alignItems: "center",
            justifyContent: "center",
            gap: "0.6rem",
            mb: "1.15rem",
            fontSize: "0.72rem",
            fontWeight: 600,
            letterSpacing: "0.14em",
            textTransform: "uppercase",
            color: "rgba(255, 255, 255, 0.78)",
            ...row(0.08),
            "& span:not(:last-child)::after": {
              content: '"\u00b7"',
              color: ORANGE,
              pl: "0.6em",
            },
          }}
        >
          <span>MuJoCo physics</span>
          <span>ONNX policies at 50 Hz</span>
        </Box>

        {/* Landing-grade h1: same type conventions as the vitrine hero. */}
        <Typography
          component="h1"
          sx={{
            m: "0 auto",
            maxWidth: "24ch",
            fontSize: { xs: "2rem", sm: "clamp(2.4rem, 5vw, 3.75rem)" },
            fontWeight: 600,
            letterSpacing: "-0.03em",
            color: "#fff",
            lineHeight: 1.02,
            textWrap: "balance",
            ...row(0.16),
          }}
        >
          The Microduck simulator, live in your browser
        </Typography>
        <Typography
          sx={{
            mx: "auto",
            mt: "1.25rem",
            mb: 0,
            "@media (max-height: 700px)": { mt: "0.85rem" },
            maxWidth: "36ch",
            fontSize: { xs: "0.95rem", sm: "clamp(1.05rem, 1.5vw, 1.3rem)" },
            lineHeight: 1.5,
            letterSpacing: "-0.012em",
            color: "rgba(255, 255, 255, 0.72)",
            textWrap: "balance",
            ...row(0.24),
          }}
        >
          The exact same trained policies that drive the real robot.
        </Typography>

        <Box
          sx={{
            display: "grid",
            gridTemplateColumns: {
              xs: variant === "touch" ? "repeat(3, 1fr)" : "repeat(2, 1fr)",
              sm: "repeat(3, 1fr)",
            },
            gap: "0.55rem",
            m: "1.4rem auto 0",
            maxWidth: "36rem",
            ...row(0.32),
          }}
        >
          {tiles.map((t) => (
            <Box
              key={t.name}
              sx={{
                p: { xs: "0.65rem 0.35rem 0.6rem", sm: "0.85rem 0.45rem 0.75rem" },
                border: "1px solid rgba(255, 255, 255, 0.08)",
                borderRadius: "18px",
                background: "rgba(255, 255, 255, 0.035)",
              }}
            >
              <Box
                sx={{
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  justifyContent: "center",
                  gap: "0.2rem",
                  height: { xs: "2.8rem", sm: "3.3rem" },
                  // Arrow cluster keycaps are a notch smaller so two rows fit
                  "& .cluster kbd": {
                    minWidth: "1.4rem",
                    height: "1.4rem",
                    p: "0 0.3rem",
                    fontSize: "0.62rem",
                  },
                }}
              >
                {t.caps === "cluster-arrows" ? (
                  <Box
                    className="cluster"
                    sx={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "0.2rem" }}
                  >
                    <Box sx={{ display: "flex", gap: "0.22rem", justifyContent: "center" }}>
                      <Kbd>{"\u2191"}</Kbd>
                    </Box>
                    <Box sx={{ display: "flex", gap: "0.22rem", justifyContent: "center" }}>
                      <Kbd>{"\u2190"}</Kbd>
                      <Kbd>{"\u2193"}</Kbd>
                      <Kbd>{"\u2192"}</Kbd>
                    </Box>
                  </Box>
                ) : (
                  <Box sx={{ display: "flex", gap: "0.22rem", justifyContent: "center" }}>
                    {t.caps.map((c) => (
                      <Kbd key={c} round={t.round ? 1 : 0}>{c}</Kbd>
                    ))}
                  </Box>
                )}
              </Box>
              <Box
                sx={{
                  mt: "0.55rem",
                  fontSize: "0.72rem",
                  fontWeight: 600,
                  letterSpacing: "-0.01em",
                  color: "#fff",
                }}
              >
                {t.name}
              </Box>
              <Box
                sx={{
                  mt: "0.18rem",
                  fontSize: "0.62rem",
                  fontWeight: 600,
                  letterSpacing: "0.12em",
                  textTransform: "uppercase",
                  color: ORANGE,
                }}
              >
                {t.hint}
              </Box>
            </Box>
          ))}
        </Box>

        <Typography
          sx={{
            mt: "1rem",
            fontSize: "0.68rem",
            fontWeight: 600,
            letterSpacing: "0.08em",
            color: "rgba(255, 255, 255, 0.4)",
            ...row(0.48),
          }}
        >
          {HINTS[variant]}
        </Typography>
      </Box>

      <Box
        sx={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: "1rem",
          mt: "1.6rem",
          "@media (max-height: 700px)": { mt: "1.1rem" },
          width: "100%",
          ...row(0.56),
        }}
      >
        <ButtonBase
          data-cta="1"
          onClick={closeMenu}
          focusRipple
          sx={{
            border: `3px solid ${INK}`,
            background: ORANGE,
            color: INK,
            font: "inherit",
            fontSize: { xs: "0.95rem", sm: "1.02rem" },
            fontWeight: 600,
            letterSpacing: "-0.01em",
            p: { xs: "0.75rem 1.5rem", sm: "0.85rem 1.9rem" },
            borderRadius: "999px",
            boxShadow: "0 0 0 3px #fff",
            cursor: "pointer",
            transition: "transform 0.2s ease, box-shadow 0.2s ease",
            "&:hover": {
              transform: "translateY(-2px)",
              boxShadow: "0 0 0 3px #fff, 0 10px 26px rgba(255, 122, 47, 0.3)",
            },
            "&:active": {
              transform: "translateY(1px)",
              boxShadow: "0 0 0 3px #fff",
            },
            "&.Mui-focusVisible": {
              outline: "2px solid #fff",
              outlineOffset: "4px",
            },
          }}
        >
          {entered ? "Resume" : "Waddle in"}
        </ButtonBase>
      </Box>
    </Box>
  );
}
