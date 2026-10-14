import { cpSync, mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { defineConfig, type Plugin } from "vite";

const cesiumBuildRoot = fileURLToPath(
  new URL("./node_modules/cesium/Build/Cesium/", import.meta.url),
);

function copyCesiumRuntime(): Plugin {
  return {
    name: "copy-cesium-runtime",
    closeBundle() {
      const destination = fileURLToPath(new URL("./dist/cesium/", import.meta.url));
      mkdirSync(destination, { recursive: true });
      for (const directory of ["Assets", "ThirdParty", "Widgets", "Workers"]) {
        cpSync(`${cesiumBuildRoot}${directory}`, `${destination}${directory}`, {
          recursive: true,
        });
      }
    },
  };
}

export default defineConfig({
  plugins: [copyCesiumRuntime()],
  build: {
    outDir: "dist",
  },
  server: {
    fs: {
      strict: true,
    },
  },
});
