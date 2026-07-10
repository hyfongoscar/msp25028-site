import { useCallback, useEffect, useMemo, useState } from "react";
import { ChevronDown, Zap } from "lucide-react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { TEAL } from "./utils/colors";
import ModelCard from "./components/ModelCard";
import MODELS from "./utils/models";
import type { PricePoint, StockSummary } from "./types/finance";
import { fetchStockData, fetchForecast } from "./utils/finance";

const STOCKS = [
  { label: "AAPL (Apple Inc.)", value: "AAPL" },
  { label: "TSLA (Tesla, Inc.)", value: "TSLA" },
  { label: "MSFT (Microsoft Corp.)", value: "MSFT" },
  { label: "AMZN (Amazon.com Inc.)", value: "AMZN" },
  { label: "NVDA (NVIDIA Corp.)", value: "NVDA" },
  { label: "GOOGL (Alphabet Inc.)", value: "GOOGL" },
];

export default function App() {
  const [selectedStock, setSelectedStock] = useState("AAPL");
  const [activeModel, setActiveModel] = useState("qlstm");
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [running, setRunning] = useState(false);
  const [ran, setRan] = useState(false);
  const [priceData, setPriceData] = useState<PricePoint[]>([]);
  const [predictedData, setPredictedData] = useState<PricePoint[]>([]);
  const [summary, setSummary] = useState<StockSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadData = useCallback(
    async (symbol = selectedStock) => {
      setLoading(true);
      setError(null);
      setRan(false);

      try {
        const result = await fetchStockData(symbol);
        const predictions = await fetchForecast(result.points, activeModel);
        setPriceData(result.points);
        setPredictedData(predictions);
        setSummary(result.summary);
        setRan(true);
      } catch (err) {
        setPriceData([]);
        setPredictedData([]);
        setSummary(null);
        setError(
          err instanceof Error ? err.message : "Unable to load stock data.",
        );
        setRan(false);
      } finally {
        setLoading(false);
        setRunning(false);
      }
    },
    [selectedStock],
  );

  useEffect(() => {
    loadData(selectedStock);
  }, [loadData, selectedStock]);

  useEffect(() => {
    (async () => {
      setPredictedData(await fetchForecast(priceData, activeModel));
    })();
  }, [activeModel]);

  const chartData = useMemo(() => {
    if (priceData.length === 0 || predictedData.length !== priceData.length) {
      return priceData.map((point) => {
        return { date: point.date, actual: point.price, forecast: 0 };
      });
    }
    return predictedData.map((point, index) => {
      return {
        date: point.date,
        actual: point.price,
        forecast: priceData[index].price,
      };
    });
  }, [priceData, predictedData]);

  const handleRun = () => {
    setRunning(true);
    void loadData(selectedStock);
  };

  const selectedLabel =
    STOCKS.find((s) => s.value === selectedStock)?.label ?? selectedStock;
  const serviceState = loading
    ? "Loading live market data..."
    : error
      ? "API unavailable"
      : "Model Service: ONLINE";

  const activeModelTitle = useMemo(
    () =>
      MODELS.find((model) => model.id === activeModel)?.title ??
      "Quantum Model",
    [activeModel],
  );

  return (
    <div className="flex min-h-screen flex-col bg-[#101217] font-sans text-[#f0f2f5]">
      <header className="sticky top-0 z-50 flex h-14 shrink-0 items-center justify-between border-b border-white/10 bg-[#1A1F2C] px-6">
        <div className="flex items-center gap-2.5">
          <Zap size={18} color={TEAL} />
          <span className="text-[15px] font-semibold tracking-[0.02em] text-[#f0f2f5]">
            Comparative Analysis of Quantum Neural Networks in Finance
          </span>
          <span className="mx-1 text-[15px] text-white/25">|</span>
          <span className="font-mono text-[13px] font-normal uppercase tracking-[0.08em] text-[#8892a4]">
            CAPSTONE
          </span>
        </div>

        <div className="flex items-center gap-2 rounded-full border border-emerald-400/20 bg-emerald-400/10 px-3 py-1.5">
          <span className="inline-block h-2 w-2 rounded-full bg-emerald-400 shadow-[0_0_6px_#00ff88]" />
          <span className="font-mono text-[11px] font-medium uppercase tracking-wider text-emerald-400">
            {serviceState}
          </span>
        </div>
      </header>

      <div className="px-6 pb-0 pt-5">
        <h1 className="m-0 text-[22px] font-bold tracking-[-0.01em] text-[#f0f2f5]">
          Quantum Finance Predictor
        </h1>
      </div>

      <div className="grid flex-1 grid-cols-1 gap-4 px-6 py-4 lg:grid-cols-[240px_minmax(0,1fr)] lg:gap-4">
        <aside className="flex flex-col gap-4 self-start rounded-xl border border-white/10 bg-[#1A1F2C] p-5">
          <p className="m-0 font-mono text-[10px] font-bold uppercase tracking-widest text-[#8892a4]">
            Model Configuration
          </p>

          <div className="relative">
            <label className="mb-1.5 block text-[11px] text-[#8892a4]">
              Select Stock/Asset (Yahoo Finance)
            </label>
            <button
              onClick={() => setDropdownOpen((value) => !value)}
              className="flex w-full items-center justify-between rounded-lg border border-white/10 bg-[#222736] px-3 py-2.5 text-left text-[13px] text-[#f0f2f5] transition-colors hover:border-cyan-400/70"
            >
              <span className="overflow-hidden text-ellipsis whitespace-nowrap">
                {selectedLabel}
              </span>
              <ChevronDown
                size={14}
                color="#8892a4"
                className={`shrink-0 transition-transform ${dropdownOpen ? "rotate-180" : "rotate-0"}`}
              />
            </button>

            {dropdownOpen && (
              <div className="absolute left-0 right-0 top-[calc(100%+4px)] z-50 overflow-hidden rounded-lg border border-white/10 bg-[#222736] shadow-[0_8px_24px_rgba(0,0,0,0.4)]">
                {STOCKS.map((stock) => (
                  <button
                    key={stock.value}
                    onClick={() => {
                      setSelectedStock(stock.value);
                      setDropdownOpen(false);
                    }}
                    className={`block w-full px-3 py-2.5 text-left text-[13px] ${stock.value === selectedStock ? "bg-cyan-400/10 text-cyan-400" : "text-[#f0f2f5] hover:bg-white/5"}`}
                  >
                    {stock.label}
                  </button>
                ))}
              </div>
            )}
          </div>

          <button
            onClick={handleRun}
            disabled={running}
            className={`flex items-center justify-center gap-1.5 rounded-lg px-0 py-2.5 text-[13px] font-bold tracking-[0.02em] text-[#101217] transition-colors ${running ? "cursor-not-allowed bg-cyan-400/30" : "bg-cyan-400 hover:bg-cyan-300"}`}
          >
            <Zap size={14} />
            {running ? "Running..." : "Run Inference"}
          </button>

          {ran && !error && (
            <p className="m-0 text-[11px] leading-5 text-emerald-400">
              ✓ Inference complete. Chart updated from the live market feed.
            </p>
          )}

          {error && (
            <p className="m-0 text-[11px] leading-5 text-amber-300">{error}</p>
          )}

          <p className="m-0 text-[11px] leading-6 text-[#8892a4]">
            Pull the latest daily prices from Yahoo Finance and generate fresh
          </p>

          <div className="mt-1 border-t border-white/10 pt-3.5">
            <p className="mb-2 font-mono text-[10px] uppercase tracking-[0.08em] text-[#8892a4]">
              Active Model
            </p>
            <p className="m-0 text-[12px] font-semibold text-cyan-400">
              {activeModelTitle}
            </p>
          </div>
        </aside>

        <div className="flex min-w-0 flex-col gap-4">
          <div>
            <p className="m-0 font-mono text-[10px] font-bold uppercase tracking-widest text-[#8892a4]">
              Comparison Dashboard
            </p>
          </div>

          <div className="flex flex-wrap gap-3">
            {MODELS.map((model) => (
              <ModelCard
                key={model.id}
                model={model}
                active={activeModel === model.id}
                onClick={() => setActiveModel(model.id)}
                trueSeries={priceData}
                predictedSeries={predictedData}
              />
            ))}
          </div>

          <div className="flex-1 rounded-xl border border-white/10 bg-[#1A1F2C] p-5">
            <div className="mb-4 flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
              <div>
                <p className="m-0 mb-0.5 text-[13px] font-semibold text-[#f0f2f5]">
                  Price Forecast — {selectedStock}
                </p>
                <p className="m-0 text-[11px] text-[#8892a4]">
                  {activeModelTitle} ·{" "}
                  {summary
                    ? `$${summary.latest.toFixed(2)} latest`
                    : "Live data pending"}
                </p>
              </div>
              <div className="flex flex-wrap items-center gap-4">
                <span className="flex items-center gap-1.5 text-[11px] text-[#8892a4]">
                  <span className="inline-block h-0.5 w-5 rounded-full bg-white" />
                  Actual Price
                </span>
                <span className="flex items-center gap-1.5 text-[11px] text-[#8892a4]">
                  <span className="inline-block h-0.5 w-5 rounded-full bg-cyan-400" />
                  Model Forecast
                </span>
              </div>
            </div>

            {loading && !priceData.length ? (
              <div className="flex h-57.5 items-center justify-center rounded-lg border border-dashed border-white/10 text-[12px] text-[#8892a4]">
                Loading live market data...
              </div>
            ) : error && !priceData.length ? (
              <div className="flex h-57.5 items-center justify-center rounded-lg border border-dashed border-white/10 text-[12px] text-amber-300">
                {error}
              </div>
            ) : (
              <>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart
                    data={chartData}
                    margin={{ top: 4, right: 8, left: 0, bottom: 0 }}
                  >
                    <CartesianGrid
                      strokeDasharray="3 3"
                      stroke="rgba(255,255,255,0.05)"
                    />
                    <XAxis
                      dataKey="date"
                      tick={{
                        fill: "#8892a4",
                        fontSize: 10,
                        fontFamily: "JetBrains Mono, monospace",
                      }}
                      tickLine={false}
                      axisLine={{ stroke: "rgba(255,255,255,0.08)" }}
                      interval={9}
                    />
                    <YAxis
                      tick={{
                        fill: "#8892a4",
                        fontSize: 10,
                        fontFamily: "JetBrains Mono, monospace",
                      }}
                      tickLine={false}
                      axisLine={{ stroke: "rgba(255,255,255,0.08)" }}
                      tickFormatter={(value) => `$${value}`}
                      label={{
                        value: "Price (USD)",
                        angle: -90,
                        position: "insideLeft",
                        offset: 12,
                        style: {
                          fill: "#8892a4",
                          fontSize: 10,
                          fontFamily: "Inter, sans-serif",
                        },
                      }}
                      width={60}
                      domain={["auto", "auto"]}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "#222736",
                        border: "1px solid rgba(255,255,255,0.12)",
                        borderRadius: 12,
                      }}
                      labelStyle={{ color: "#f0f2f5" }}
                      formatter={(value) => [
                        `$${Number(value).toFixed(2)}`,
                        "Price",
                      ]}
                    />
                    <Legend wrapperStyle={{ display: "none" }} />
                    <Line
                      type="monotone"
                      dataKey="actual"
                      name="Actual Price"
                      stroke="rgba(255,255,255,0.7)"
                      strokeWidth={1.5}
                      dot={false}
                      activeDot={{ r: 4, fill: "#fff" }}
                    />
                    <Line
                      type="monotone"
                      dataKey="forecast"
                      name="Model Forecast"
                      stroke={TEAL}
                      strokeWidth={1.5}
                      dot={false}
                      activeDot={{ r: 4, fill: TEAL }}
                    />
                  </LineChart>
                </ResponsiveContainer>

                <div className="mt-1 text-center">
                  <span className="font-mono text-[10px] text-[#8892a4]">
                    Date
                  </span>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
