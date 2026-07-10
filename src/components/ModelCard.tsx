import { useMemo } from "react";
import {
  Bar,
  BarChart,
  Line,
  LineChart as RechartsLineChart,
  ResponsiveContainer,
} from "recharts";
import type MODELS from "../utils/models";
import { PURPLE, TEAL } from "../utils/colors";
import type { PricePoint, StockSummary } from "../types/finance";

type Props = {
  model: (typeof MODELS)[0];
  active: boolean;
  onClick: () => void;
  trueSeries: PricePoint[];
  predictedSeries: PricePoint[];
};

function ModelCard({
  model,
  active,
  onClick,
  trueSeries,
  predictedSeries,
}: Props) {
  const sparkData = useMemo(() => {
    if (
      predictedSeries.length === 0 ||
      predictedSeries.length !== trueSeries.length
    ) {
      return trueSeries.map((point, index) => {
        return { i: index, a: point.price, b: 0 };
      });
    }
    return predictedSeries.map((point, index) => {
      return { i: index, a: point.price, b: trueSeries[index].price };
    });
  }, [trueSeries]);

  return (
    <button
      onClick={onClick}
      className={`flex-1 min-w-0 rounded-[10px] border px-4 py-4 text-left transition-all duration-200 focus:outline-none ${active ? "border-cyan-400/80 bg-cyan-400/10 shadow-[0_0_20px_rgba(0,242,254,0.12)]" : "border-white/10 bg-[#1A1F2C]"}`}
    >
      <p
        className={`mb-3 font-mono text-[11px] font-semibold uppercase tracking-[0.08em] ${active ? "text-cyan-400" : "text-[#8892a4]"}`}
      >
        {model.title}
      </p>

      <div className="mt-2 h-12">
        {model.vizType === "bar" ? (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={sparkData} barGap={1}>
              <Bar dataKey="a" fill={TEAL} radius={[2, 2, 0, 0]} />
              <Bar dataKey="b" fill={PURPLE} radius={[2, 2, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <RechartsLineChart data={sparkData}>
              <Line
                type="monotone"
                dataKey="a"
                stroke={TEAL}
                strokeWidth={1.5}
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="b"
                stroke={PURPLE}
                strokeWidth={1.5}
                dot={false}
              />
            </RechartsLineChart>
          </ResponsiveContainer>
        )}
      </div>
    </button>
  );
}

export default ModelCard;
