import { useCallback, useEffect, useMemo, useState } from 'react';

import { ChevronDown, Zap } from 'lucide-react';
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

import ModelCard from './components/ModelCard';
import type {
  PricePoint,
  StockPricePoint,
  StockSummary,
} from './types/finance';
import { TEAL } from './utils/colors';
import { fetchForecast, fetchStockData } from './utils/finance';
import MODELS from './utils/models';

// Expanded Asset Directory categorized by class
const ASSETS = [
  // Equities
  { label: 'AAPL (Apple Inc.)', value: 'AAPL', group: 'Stocks' },
  { label: 'TSLA (Tesla, Inc.)', value: 'TSLA', group: 'Stocks' },
  { label: 'MSFT (Microsoft Corp.)', value: 'MSFT', group: 'Stocks' },
  { label: 'AMZN (Amazon.com Inc.)', value: 'AMZN', group: 'Stocks' },
  { label: 'NVDA (NVIDIA Corp.)', value: 'NVDA', group: 'Stocks' },
  { label: 'GOOGL (Alphabet Inc.)', value: 'GOOGL', group: 'Stocks' },
  // Cryptocurrencies (yfinance format)
  { label: 'BTC-USD (Bitcoin)', value: 'BTC-USD', group: 'Crypto' },
  { label: 'ETH-USD (Ethereum)', value: 'ETH-USD', group: 'Crypto' },
  { label: 'SOL-USD (Solana)', value: 'SOL-USD', group: 'Crypto' },
  { label: 'XRP-USD (Ripple)', value: 'XRP-USD', group: 'Crypto' },
];

const defaultPredictions: Record<string, PricePoint[]> = {
  lstm: [],
  qlstm: [],
  custom_qnn: [],
  hybrid_qnn1: [],
  hybrid_qnn2: [],
  hybrid_qnn1_binary: [],
  hybrid_qnn2_binary: [],
};

export default function App() {
  const [selectedAsset, setSelectedAsset] = useState('AAPL');
  const [activeModel, setActiveModel] = useState('lstm');

  const [loadedAsset, setLoadedAsset] = useState<string | null>(null);
  const [priceData, setPriceData] = useState<StockPricePoint[]>([]);
  const [predictedData, setPredictedData] = useState(defaultPredictions);
  const [predictedStats, setPredictedStats] = useState<
    Record<string, Record<string, number>>
  >({});
  const [summary, setSummary] = useState<StockSummary | null>(null);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [running, setRunning] = useState(false);
  const [ran, setRan] = useState(false);

  // Check if current view is a classification variant
  const isBinaryModel = useMemo(
    () => activeModel.endsWith('_binary'),
    [activeModel],
  );

  const loadData = useCallback(async (symbol: string) => {
    setLoading(true);
    setError(null);
    setRan(false);

    try {
      const result = await fetchStockData(symbol);
      const forecast = await fetchForecast(result.points);
      setLoadedAsset(symbol);
      setPriceData(result.points);
      setSummary(result.summary);
      setPredictedData(forecast.predictions);
      setPredictedStats(forecast.stats);
      setRan(true);
    } catch (err) {
      setLoadedAsset(null);
      setPriceData([]);
      setSummary(null);
      setPredictedData(defaultPredictions);
      setPredictedStats({});
      setError(
        err instanceof Error ? err.message : 'Unable to load asset data.',
      );
      setRan(false);
    } finally {
      setLoading(false);
      setRunning(false);
    }
  }, []);

  useEffect(() => {
    (async () => {
      void loadData('AAPL');
    })();
  }, [loadData]);

  const chartData = useMemo(() => {
    const predictions = predictedData[activeModel] || [];
    const validPriceData = priceData.filter(point => !!point.close);

    if (
      validPriceData.length === 0 ||
      predictions.length !== validPriceData.length
    ) {
      return validPriceData.map(point => ({
        date: point.date,
        actual: point.close,
        forecast: 0,
        probability: isBinaryModel ? 50 : null,
      }));
    }

    return predictions.map((point, index) => {
      const actualPrice = priceData[index].close;

      return {
        date: point.date || priceData[index].date,
        actual: actualPrice,
        forecast: point.price !== undefined ? point.price : null,
        probability:
          point.probability !== undefined ? point.probability * 100 : null,
      };
    });
  }, [predictedData, activeModel, priceData, isBinaryModel]);

  // Derived classification insights for quick dashboard reporting
  const binaryStats = useMemo(() => {
    if (!isBinaryModel || chartData.length === 0) return null;
    const upSignals = chartData.filter(d => d.forecast === 1).length;
    const downSignals = chartData.filter(d => d.forecast === 0).length;
    const avgConfidence =
      chartData.reduce((acc, curr) => acc + (curr.probability || 0), 0) /
      chartData.length;

    return { upSignals, downSignals, avgConfidence };
  }, [chartData, isBinaryModel]);

  const handleRun = () => {
    setRunning(true);
    void loadData(selectedAsset);
  };

  const selectedLabel =
    ASSETS.find(s => s.value === selectedAsset)?.label ?? selectedAsset;
  const serviceState = loading
    ? 'Loading live market data...'
    : error
      ? 'API unavailable'
      : 'Model Service: ONLINE';

  const activeModelTitle = useMemo(
    () =>
      MODELS.find(model => model.id === activeModel)?.title ?? 'Quantum Model',
    [activeModel],
  );

  return (
    <div className="flex min-h-screen flex-col bg-[#101217] font-sans text-[#f0f2f5]">
      <header className="sticky top-0 z-50 flex h-14 shrink-0 items-center justify-between border-b border-white/10 bg-[#1A1F2C] px-6">
        <div className="flex items-center gap-2.5">
          <Zap color={TEAL} size={18} />
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
              className="flex w-full items-center justify-between rounded-lg border border-white/10 bg-[#222736] px-3 py-2.5 text-left text-[13px] text-[#f0f2f5] transition-colors hover:border-cyan-400/70"
              onClick={() => setDropdownOpen(value => !value)}
            >
              <span className="overflow-hidden text-ellipsis whitespace-nowrap">
                {selectedLabel}
              </span>
              <ChevronDown
                className={`shrink-0 transition-transform ${dropdownOpen ? 'rotate-180' : 'rotate-0'}`}
                color="#8892a4"
                size={14}
              />
            </button>

            {dropdownOpen && (
              <div className="absolute left-0 right-0 top-[calc(100%+4px)] z-50 max-h-96 overflow-y-auto rounded-lg border border-white/10 bg-[#222736] shadow-[0_8px_24px_rgba(0,0,0,0.4)] scrollbar-thin scrollbar-thumb-white/10">
                {/* Equities Category */}
                <div className="px-3 py-1.5 text-[10px] font-bold uppercase tracking-wider text-[#8892a4] border-b border-white/5 bg-[#1e2230]">
                  Equities
                </div>
                {ASSETS.filter(asset => asset.group === 'Stocks').map(asset => (
                  <button
                    className={`block w-full px-3 py-2 text-left text-[13px] ${asset.value === selectedAsset ? 'bg-cyan-400/10 text-cyan-400 font-medium' : 'text-[#f0f2f5] hover:bg-white/5'}`}
                    key={asset.value}
                    onClick={() => {
                      setSelectedAsset(asset.value);
                      setDropdownOpen(false);
                    }}
                  >
                    {asset.label}
                  </button>
                ))}

                {/* Crypto Category */}
                <div className="px-3 py-1.5 text-[10px] font-bold uppercase tracking-wider text-[#8892a4] border-t border-b border-white/5 bg-[#1e2230]">
                  Cryptocurrencies
                </div>
                {ASSETS.filter(asset => asset.group === 'Crypto').map(asset => (
                  <button
                    className={`block w-full px-3 py-2 text-left text-[13px] ${asset.value === selectedAsset ? 'bg-cyan-400/10 text-cyan-400 font-medium' : 'text-[#f0f2f5] hover:bg-white/5'}`}
                    key={asset.value}
                    onClick={() => {
                      setSelectedAsset(asset.value);
                      setDropdownOpen(false);
                    }}
                  >
                    {asset.label}
                  </button>
                ))}
              </div>
            )}
          </div>

          <button
            className={`flex items-center justify-center gap-1.5 rounded-lg px-0 py-2.5 text-[13px] font-bold tracking-[0.02em] text-[#101217] transition-colors ${running ? 'cursor-not-allowed bg-cyan-400/30' : 'bg-cyan-400 hover:bg-cyan-300'}`}
            disabled={running}
            onClick={handleRun}
          >
            <Zap size={14} />
            {running ? 'Running...' : 'Run Inference'}
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
            Pull the latest daily prices from Yahoo Finance and generate
            predictions.
          </p>
        </aside>

        <div className="flex min-w-0 flex-col gap-4">
          <div>
            <p className="m-0 font-mono text-[10px] font-bold uppercase tracking-widest text-[#8892a4]">
              Comparison Dashboard
            </p>
          </div>

          <div className="grid grid-cols-4 gap-3">
            {MODELS.map(model => (
              <ModelCard
                active={activeModel === model.id}
                key={model.id}
                model={model}
                onClick={() => setActiveModel(model.id)}
                predictedStats={predictedStats}
              />
            ))}
          </div>

          {/* Classification Stats Overlay Section */}
          {isBinaryModel && binaryStats && (
            <div className="grid grid-cols-3 gap-4 rounded-xl border border-white/10 bg-[#1A1F2C] p-4 font-mono text-xs">
              <div>
                <span className="text-[#8892a4]">UP CALLS (▲):</span>
                <span className="ml-2 font-bold text-emerald-400">
                  {binaryStats.upSignals} days
                </span>
              </div>
              <div>
                <span className="text-[#8892a4]">DOWN CALLS (▼):</span>
                <span className="ml-2 font-bold text-rose-400">
                  {binaryStats.downSignals} days
                </span>
              </div>
              <div>
                <span className="text-[#8892a4]">AVG UP PROBABILITY:</span>
                <span className="ml-2 font-bold text-cyan-400">
                  {binaryStats.avgConfidence.toFixed(1)}%
                </span>
              </div>
            </div>
          )}

          <div className="flex-1 rounded-xl border border-white/10 bg-[#1A1F2C] p-5">
            <div className="mb-4 flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
              <div>
                <p className="m-0 mb-0.5 text-[13px] font-semibold text-[#f0f2f5]">
                  {isBinaryModel
                    ? 'Directional Probability Matrix'
                    : 'Price Forecast'}{' '}
                  — {loadedAsset}
                </p>
                <p className="m-0 text-[11px] text-[#8892a4]">
                  {activeModelTitle} ·{' '}
                  {summary
                    ? `$${summary.latest?.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })} latest close`
                    : 'Live data pending'}
                </p>
              </div>
              <div className="flex flex-wrap items-center gap-4">
                <span className="flex items-center gap-1.5 text-[11px] text-[#8892a4]">
                  <span className="inline-block h-0.5 w-5 rounded-full bg-white" />
                  Actual Price
                </span>
                <span className="flex items-center gap-1.5 text-[11px] text-[#8892a4]">
                  <span
                    className={`inline-block h-0.5 w-5 rounded-full ${isBinaryModel ? 'bg-cyan-500/40' : 'bg-cyan-400'}`}
                  />
                  {isBinaryModel
                    ? 'Upward Trend Probability (%)'
                    : 'Model Forecast'}
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
                <ResponsiveContainer height={300} width="100%">
                  <ComposedChart
                    data={chartData}
                    margin={{ top: 4, right: 8, left: 0, bottom: 0 }}
                  >
                    <defs>
                      <linearGradient
                        id="probGradient"
                        x1="0"
                        x2="0"
                        y1="0"
                        y2="1"
                      >
                        <stop offset="5%" stopColor={TEAL} stopOpacity={0.25} />
                        <stop offset="95%" stopColor={TEAL} stopOpacity={0.0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid
                      stroke="rgba(255,255,255,0.05)"
                      strokeDasharray="3 3"
                    />
                    <XAxis
                      axisLine={{ stroke: 'rgba(255,255,255,0.08)' }}
                      dataKey="date"
                      interval={9}
                      tick={{
                        fill: '#8892a4',
                        fontSize: 10,
                        fontFamily: 'JetBrains Mono, monospace',
                      }}
                      tickLine={false}
                    />

                    {/* Primary Left Y Axis for asset dollar value tracking */}
                    <YAxis
                      axisLine={{ stroke: 'rgba(255,255,255,0.08)' }}
                      domain={['auto', 'auto']}
                      label={{
                        value: 'Asset Value (USD)',
                        angle: -90,
                        position: 'insideLeft',
                        offset: 12,
                        style: {
                          fill: '#8892a4',
                          fontSize: 10,
                          fontFamily: 'Inter, sans-serif',
                        },
                      }}
                      tick={{
                        fill: '#8892a4',
                        fontSize: 10,
                        fontFamily: 'JetBrains Mono, monospace',
                      }}
                      tickFormatter={value =>
                        `$${value.toLocaleString(undefined, { maximumFractionDigits: 0 })}`
                      }
                      tickLine={false}
                      width={65}
                      yAxisId="left"
                    />

                    {/* Secondary Right Y Axis rendered strictly for Classification tasks */}
                    {isBinaryModel && (
                      <YAxis
                        axisLine={{ stroke: 'rgba(255,255,255,0.08)' }}
                        domain={[0, 100]}
                        orientation="right"
                        tick={{
                          fill: '#8892a4',
                          fontSize: 10,
                          fontFamily: 'JetBrains Mono, monospace',
                        }}
                        tickFormatter={value => `${value}%`}
                        width={45}
                        yAxisId="right"
                      />
                    )}

                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#222736',
                        border: '1px solid rgba(255,255,255,0.12)',
                        borderRadius: 12,
                      }}
                      formatter={(value, name) => {
                        if (name === 'Upward Probability')
                          return [`${Number(value).toFixed(1)}%`, name];
                        if (name === 'Directional Call')
                          return [
                            Number(value) === 1 ? 'UP (▲)' : 'DOWN (▼)',
                            name,
                          ];
                        return [
                          `$${Number(value).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`,
                          name,
                        ];
                      }}
                      labelStyle={{ color: '#f0f2f5' }}
                    />

                    <Legend wrapperStyle={{ display: 'none' }} />

                    {/* Reference line marking the decision threshold at 50% */}
                    {isBinaryModel && (
                      <ReferenceLine
                        stroke="rgba(255, 255, 255, 0.15)"
                        strokeDasharray="4 4"
                        y={50}
                        yAxisId="right"
                      />
                    )}

                    {/* Baseline Price Tracker */}
                    <Line
                      activeDot={{ r: 4, fill: '#fff' }}
                      dataKey="actual"
                      dot={false}
                      name="Actual Price"
                      stroke="rgba(255,255,255,0.7)"
                      strokeWidth={1.5}
                      type="monotone"
                      yAxisId="left"
                    />

                    {/* Conditional rendering depending on model state */}
                    {isBinaryModel ? (
                      <>
                        <Area
                          dataKey="probability"
                          fill="url(#probGradient)"
                          name="Upward Probability"
                          opacity={1}
                          stroke={TEAL}
                          strokeWidth={1.5}
                          type="monotone"
                          yAxisId="right"
                        />
                        {/* Hidden property pass to deliver metrics into the tooltip pipeline */}
                        <Line
                          activeDot={false}
                          dataKey="prediction"
                          dot={false}
                          name="Directional Call"
                          stroke="transparent"
                          yAxisId="right"
                        />
                      </>
                    ) : (
                      <Line
                        activeDot={{ r: 4, fill: TEAL }}
                        dataKey="forecast"
                        dot={false}
                        name="Model Forecast"
                        stroke={TEAL}
                        strokeWidth={1.5}
                        type="monotone"
                        yAxisId="left"
                      />
                    )}
                  </ComposedChart>
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
