import { createRoot } from "react-dom/client";
import { ThemeProvider } from "@mui/material/styles";
import CssBaseline from "@mui/material/CssBaseline";
import { theme } from "./theme.js";
import App from "./App.jsx";

// No StrictMode: the game core is a heavyweight singleton (MuJoCo WASM,
// ONNX sessions, WebRTC room) and dev double-mounting would double-boot it.
createRoot(document.getElementById("root")).render(
  <ThemeProvider theme={theme}>
    <CssBaseline />
    <App />
  </ThemeProvider>,
);
