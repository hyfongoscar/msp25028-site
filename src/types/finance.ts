export type PricePoint = {
  date: string;
  actual: number;
  forecast: number;
};

export type StockSummary = {
  latest: number;
  change: number;
  volatility: number;
  trend: number;
};

export type StockPayload = {
  points: PricePoint[];
  summary: StockSummary;
};
