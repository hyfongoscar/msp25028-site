export type PricePoint = {
  date: string;
  price: number;
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
