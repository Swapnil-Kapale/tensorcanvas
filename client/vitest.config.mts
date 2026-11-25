import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

const clientDir = fileURLToPath(new URL(".", import.meta.url));

export default defineConfig({
  resolve: {
    // Same "@/*" mapping as tsconfig.json, so tests can import "@/lib/..." like the app.
    alias: { "@": path.resolve(clientDir) },
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./vitest.setup.ts"],
  },
});
