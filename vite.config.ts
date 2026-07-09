import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import YahooFinance from "yahoo-finance2";

const yahooFinance = new YahooFinance();

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

          const url = new URL(req.url, "http://localhost");
          const symbol = url.searchParams.get("symbol") ?? "AAPL";
          const range =
            (url.searchParams.get("range") as
              | "6mo"
              | "1y"
              | "2y"
              | "5y"
              | "max") ?? "6mo";
          const interval =
            (url.searchParams.get("interval") as "1d" | "1wk" | "1mo") ?? "1d";
          const endDate = new Date();
          const startDate = new Date(endDate);

          switch (range) {
            case "6mo":
              startDate.setMonth(startDate.getMonth() - 6);
              break;
            case "1y":
              startDate.setFullYear(startDate.getFullYear() - 1);
              break;
            case "2y":
              startDate.setFullYear(startDate.getFullYear() - 2);
              break;
            case "5y":
              startDate.setFullYear(startDate.getFullYear() - 5);
              break;
            case "max":
              startDate.setFullYear(startDate.getFullYear() - 30);
              break;
          }

          try {
            const result = await yahooFinance.chart(symbol, {
              period1: startDate,
              period2: endDate,
              interval,
            });
            res.statusCode = 200;
            res.setHeader("Content-Type", "application/json");
            res.end(JSON.stringify(result));
          } catch (error) {
            res.statusCode = 500;
            res.setHeader("Content-Type", "application/json");
            res.end(
              JSON.stringify({
                error: error instanceof Error ? error.message : String(error),
              }),
            );
          }
        });
      },
    },
  ],
  server: {},
});
