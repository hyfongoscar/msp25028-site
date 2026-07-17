import type MODELS from '../utils/models';

type Props = {
  model: (typeof MODELS)[0];
  active: boolean;
  onClick: () => void;
  predictedStats: Record<string, Record<string, number>>;
};

function ModelCard({ model, active, onClick, predictedStats }: Props) {
  return (
    <button
      className={`flex-1 min-w-0 rounded-[10px] border px-4 py-4 flex flex-col justify-start items-stretch transition-all duration-200 focus:outline-none ${active ? 'border-cyan-400/80 bg-cyan-400/10 shadow-[0_0_20px_rgba(0,242,254,0.12)]' : 'border-white/10 bg-[#1A1F2C]'}`}
      onClick={onClick}
    >
      <p
        className={`mb-3 w-full text-left font-mono text-[11px] font-semibold uppercase tracking-[0.08em] ${active ? 'text-cyan-400' : 'text-[#8892a4]'}`}
      >
        {model.title}
      </p>
      <div className="flex gap-4">
        <div className="mb-3 space-y-1 flex-1">
          <div className="text-[12px] font-semibold">Validation Stats</div>
          {!!model.accuracy && (
            <div className="flex items-baseline justify-between">
              <span className="text-[10px] text-[#8892a4]">Accuracy</span>
              <span className="font-mono text-[12px] font-semibold text-cyan-400">
                {model.accuracy}
              </span>
            </div>
          )}
          {!!model.rmse && (
            <div className="flex items-baseline justify-between">
              <span className="text-[10px] text-[#8892a4]">RMSE</span>
              <span className="font-mono text-[12px] font-semibold text-cyan-400">
                {model.rmse}
              </span>
            </div>
          )}
          {!!model.roc_auc && (
            <div className="flex items-baseline justify-between">
              <span className="text-[10px] text-[#8892a4]">ROC-AUC</span>
              <span className="font-mono text-[12px] font-semibold text-[#f0f2f5]">
                {model.roc_auc}
              </span>
            </div>
          )}
          {!!model.mae && (
            <div className="flex items-baseline justify-between">
              <span className="text-[10px] text-[#8892a4]">MAE</span>
              <span className="font-mono text-[12px] font-semibold text-[#f0f2f5]">
                {model.mae}
              </span>
            </div>
          )}
          {!!model.r_squared && (
            <div className="flex items-baseline justify-between">
              <span className="text-[10px] text-[#8892a4]">R^2</span>
              <span className="font-mono text-[12px] font-semibold text-[#f0f2f5]">
                {model.r_squared}
              </span>
            </div>
          )}
        </div>
        <div className="w-px bg-[#666666]" />
        <div className="mb-3 space-y-1 flex-1">
          <div className="text-[12px] font-semibold">Live Stats</div>
          {!!model.accuracy && (
            <div className="flex items-baseline justify-between">
              <span className="text-[10px] text-[#8892a4]">Accuracy</span>
              <span className="font-mono text-[12px] font-semibold text-cyan-400">
                {predictedStats[model.id]?.accuracy
                  ? (predictedStats[model.id].accuracy * 100)?.toFixed(2) + '%'
                  : '-'}
              </span>
            </div>
          )}
          {!!model.rmse && (
            <div className="flex items-baseline justify-between">
              <span className="text-[10px] text-[#8892a4]">RMSE</span>
              <span className="font-mono text-[12px] font-semibold text-cyan-400">
                {predictedStats[model.id]?.rmse?.toFixed(4) || '-'}
              </span>
            </div>
          )}
          {!!model.roc_auc && (
            <div className="flex items-baseline justify-between">
              <span className="text-[10px] text-[#8892a4]">ROC-AUC</span>
              <span className="font-mono text-[12px] font-semibold text-[#f0f2f5]">
                {predictedStats[model.id]?.roc_auc?.toFixed(4) || '-'}
              </span>
            </div>
          )}
          {!!model.mae && (
            <div className="flex items-baseline justify-between">
              <span className="text-[10px] text-[#8892a4]">MAE</span>
              <span className="font-mono text-[12px] font-semibold text-[#f0f2f5]">
                {predictedStats[model.id]?.mae?.toFixed(4) || '-'}
              </span>
            </div>
          )}
          {!!model.r_squared && (
            <div className="flex items-baseline justify-between">
              <span className="text-[10px] text-[#8892a4]">R^2</span>
              <span className="font-mono text-[12px] font-semibold text-[#f0f2f5]">
                {predictedStats[model.id]?.r2?.toFixed(4) || '-'}
              </span>
            </div>
          )}
        </div>
      </div>
    </button>
  );
}

export default ModelCard;
