import type { ChartResultArray } from 'yahoo-finance2/modules/chart';

import type { PricePoint, StockPayload } from '../types/finance';

const configuredApiBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim();
const apiBaseUrl = configuredApiBaseUrl
  ? configuredApiBaseUrl
  : `${window.location.origin}/api`;

async function fetchForecast(
  prices: PricePoint[],
): Promise<Record<string, PricePoint[]>> {
  const response = await fetch(
    `${apiBaseUrl.replace(/\/$/, '')}/predict?&prices=[${prices.map(s => s.price).join(',')}]`,
  );
  if (!response.ok) {
    throw new Error('Unable to reach the predict endpoint right now.');
  }

  const result = (await response.json()) as {
    predictions: Record<string, number[]>;
  };

  const formattedPredictions: Record<string, PricePoint[]> = {};
  for (const [model, predictions] of Object.entries(result.predictions)) {
    formattedPredictions[model.toLowerCase()] = predictions.map((p, i) => ({
      date: prices[i].date,
      price: p,
    }));
  }
  return formattedPredictions;
}

async function fetchStockData(symbol: string): Promise<StockPayload> {
  const response = await fetch(
    `${apiBaseUrl.replace(/\/$/, '')}/finance?symbol=${encodeURIComponent(symbol)}&interval=1d&range=6mo`,
  );

  if (!response.ok) {
    throw new Error('Unable to reach the finance endpoint right now.');
  }

  const result = (await response.json()) as ChartResultArray;

  if (!result || !Array.isArray(result.quotes)) {
    throw new Error('No stock data was returned from Yahoo Finance.');
  }

  const points: PricePoint[] = result.quotes
    .map(quote => {
      const close = quote.close;
      return close == null
        ? null
        : {
            date: new Date(quote.date).toLocaleDateString('en-US', {
              month: 'short',
              day: 'numeric',
            }),
            price: Number(close.toFixed(2)),
          };
    })
    .filter((entry): entry is PricePoint => entry !== null)
    .slice(-60);

  if (!points.length) {
    throw new Error('No historical points were returned from Yahoo Finance.');
  }

  const latest = points[points.length - 1];
  const previous = points[points.length - 2] ?? latest;
  const change = previous.price
    ? ((latest.price - previous.price) / previous.price) * 100
    : 0;

  return {
    points: points,
    summary: {
      latest: latest.price,
      change,
      volatility:
        points.slice(-8).reduce((sum, point, index) => {
          if (index === 0) return sum;
          const previousPoint = points[points.length - 8 + index - 1];
          return sum + Math.abs(point.price - previousPoint.price);
        }, 0) / 7,
      trend: latest.price - previous.price,
    },
  };
}

export { fetchForecast, fetchStockData };
