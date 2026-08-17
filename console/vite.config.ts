import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// The console calls the API at the relative path `/answer`, so the same build works both behind
// this dev proxy and, once compose serves the built assets from the API's own origin, unchanged.
// Proxying here rather than adding CORS to the server keeps the browser talking to one origin.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      "/answer": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/questions": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  test: {
    globals: true,
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
  },
});
