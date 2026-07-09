import YahooFinance from "yahoo-finance2";

const yahooFinance = new YahooFinance();
const siteOrigin =
  process.env.SITE_URL ??
  process.env.API_BASE_URL ??
  (process.env.VERCEL_URL ? `https://${process.env.VERCEL_URL}` : "http://localhost:5173");

function getDateRange(range) {
  const endDate = new Date();
  const startDate = new Date(endDate);

  switch (range) {
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
    case "6mo":
    default:
      startDate.setMonth(startDate.getMonth() - 6);
      break;
  }

  return { startDate, endDate };
}

export default async function handler(req, res) {
  if (req.method !== "GET") {
    res.statusCode = 405;
    res.setHeader("Content-Type", "application/json");
    res.end(JSON.stringify({ error: "Method not allowed" }));
    return;
  }

  const url = new URL(req.url, siteOrigin);
  const symbol = url.searchParams.get("symbol") ?? "AAPL";
  const range = url.searchParams.get("range") ?? "6mo";
  const interval = url.searchParams.get("interval") ?? "1d";
  const { startDate, endDate } = getDateRange(range);

  try {
    const result = await yahooFinance.chart(symbol, {
      period1: startDate,
      period2: endDate,
      interval,
      return: "array",
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
}

export { handler }
