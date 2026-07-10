import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// @ts-ignore
import { handler } from "./api/finance.js";

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
    {
      name: "yfinance-api",
      configureServer(server) {
        server.middlewares.use(async (req, res, next) => {
          if (!req.url?.startsWith("/api/finance")) {
            return next();
          }

          handler(req, res);
        });
      },
    },
  ],
  server: {
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, "/api"),
      },
    },
  },
});
