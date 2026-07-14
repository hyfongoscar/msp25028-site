import { isNumber } from 'lodash-es';
import type { ChartResultArray } from 'yahoo-finance2/modules/chart';

import type {
  PricePoint,
  StockPayload,
  StockPricePoint,
} from '../types/finance';

const configuredApiBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim();
const apiBaseUrl = configuredApiBaseUrl
  ? configuredApiBaseUrl
  : `${window.location.origin}/api`;

async function fetchForecast(
  prices: StockPricePoint[],
): Promise<Record<string, PricePoint[]>> {
  const response = await fetch(
    `${apiBaseUrl.replace(/\/$/, '')}/predict?opens=[${prices.map(s => s.open).join(',')}]&highs=[${prices.map(s => s.high).join(',')}]&lows=[${prices.map(s => s.low).join(',')}]&closes=[${prices.map(s => s.close).join(',')}]&volumes=[${prices.map(s => s.volume).join(',')}]&adjCloses=[${prices.map(s => s.adjClose).join(',')}]`,
  );
  if (!response.ok) {
    throw new Error('Unable to reach the predict endpoint right now.');
  }

  const result = (await response.json()) as {
    predictions: Record<
      string,
      number[] | { probability: number; prediction: number }[]
    >;
  };

  const formattedPredictions: Record<string, PricePoint[]> = {};
  for (const [model, predictions] of Object.entries(result.predictions)) {
    formattedPredictions[model.toLowerCase()] = predictions.map((p, i) => ({
      date: prices[i].date,
      price: isNumber(p) ? p : p.prediction,
      probability: isNumber(p) ? undefined : p.probability,
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

  const points = result.quotes
    .map(quote => {
      return {
        date: new Date(quote.date).toLocaleDateString('en-US', {
          month: 'short',
          day: 'numeric',
        }),
        open: quote.open,
        high: quote.high,
        low: quote.low,
        close: quote.close,
        volume: quote.volume,
        adjClose: quote.adjclose,
      };
    })
    .slice(-60);

  if (!points.length) {
    throw new Error('No historical points were returned from Yahoo Finance.');
  }

  const latest = points[points.length - 1];
  const previous = points[points.length - 2] ?? latest;
  const trend =
    latest.close && previous.close ? latest.close - previous.close : 0;
  const change = previous.close
    ? (((latest.close || 0) - previous.close) / previous.close) * 100
    : 0;

  return {
    points: points,
    summary: {
      latest: latest.close,
      change,
      volatility:
        points.slice(-8).reduce((sum, point, index) => {
          if (index === 0) return sum;
          const previousPoint = points[points.length - 8 + index - 1];
          const difference =
            point.close && previousPoint.close
              ? Math.abs(point.close - previousPoint.close)
              : 0;
          return sum + Math.abs(difference);
        }, 0) / 7,
      trend: trend,
    },
  };
}

export { fetchForecast, fetchStockData };
