export type StockPricePoint = {
  date: string;
  open: number | null;
  high: number | null;
  low: number | null;
  close: number | null;
  volume: number | null;
  adjClose?: number | null;
};

export type PricePoint = {
  date: string;
  price: number;
};

export type StockSummary = {
  latest: number | null;
  change: number | null;
  volatility: number | null;
  trend: number | null;
};

export type StockPayload = {
  points: StockPricePoint[];
  summary: StockSummary;
};
