import type { ChartResultArray } from "yahoo-finance2/modules/chart";
import type { PricePoint, StockPayload } from "../types/finance";

function buildForecast(points: PricePoint[]) {
  const recent = points.slice(-8);
  const averageChange =
    recent.reduce((sum, point, index) => {
      if (index === 0) return sum;
      const previous = recent[index - 1];
      return sum + (point.actual - previous.actual);
    }, 0) / Math.max(recent.length - 1, 1);
  const volatility =
    recent.reduce((sum, point, index) => {
      if (index === 0) return sum;
      const previous = recent[index - 1];
      return sum + Math.abs(point.actual - previous.actual);
    }, 0) / Math.max(recent.length - 1, 1);

  return points.map((point, index) => ({
    ...point,
    forecast: Number(
      (
        point.actual +
        averageChange * 0.35 +
        (index % 2 === 0 ? 1 : -1) * (volatility / 20)
      ).toFixed(2),
    ),
  }));
}

async function fetchStockData(symbol: string): Promise<StockPayload> {
  const response = await fetch(
    `/api/finance?symbol=${encodeURIComponent(symbol)}&interval=1d&range=6mo`,
  );

  if (!response.ok) {
    throw new Error("Unable to reach the finance endpoint right now.");
  }

  const result = (await response.json()) as ChartResultArray;

  if (!result || !Array.isArray(result.quotes)) {
    throw new Error("No stock data was returned from Yahoo Finance.");
  }

  const points: PricePoint[] = result.quotes
    .map((quote) => {
      const close = quote.close;
      return close == null
        ? null
        : {
            date: new Date(quote.date).toLocaleDateString("en-US", {
              month: "short",
              day: "numeric",
            }),
            actual: Number(close.toFixed(2)),
            forecast: 0,
          };
    })
    .filter((entry): entry is PricePoint => entry !== null)
    .slice(-60);

  if (!points.length) {
    throw new Error("No historical points were returned from Yahoo Finance.");
  }

  const withForecast = buildForecast(points);
  const latest = withForecast[withForecast.length - 1];
  const previous = withForecast[withForecast.length - 2] ?? latest;
  const change = previous.actual
    ? ((latest.actual - previous.actual) / previous.actual) * 100
    : 0;

  return {
    points: withForecast,
    summary: {
      latest: latest.actual,
      change,
      volatility:
        withForecast.slice(-8).reduce((sum, point, index) => {
          if (index === 0) return sum;
          const previousPoint =
            withForecast[withForecast.length - 8 + index - 1];
          return sum + Math.abs(point.actual - previousPoint.actual);
        }, 0) / 7,
      trend: latest.actual - previous.actual,
    },
  };
}

export { fetchStockData };
