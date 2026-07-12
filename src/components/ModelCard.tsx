import type MODELS from '../utils/models';

type Props = {
  model: (typeof MODELS)[0];
  active: boolean;
  onClick: () => void;
};

function ModelCard({ model, active, onClick }: Props) {
  return (
    <button
      className={`flex-1 min-w-0 rounded-[10px] border px-4 py-4 text-left transition-all duration-200 focus:outline-none ${active ? 'border-cyan-400/80 bg-cyan-400/10 shadow-[0_0_20px_rgba(0,242,254,0.12)]' : 'border-white/10 bg-[#1A1F2C]'}`}
      onClick={onClick}
    >
      <p
        className={`mb-3 font-mono text-[11px] font-semibold uppercase tracking-[0.08em] ${active ? 'text-cyan-400' : 'text-[#8892a4]'}`}
      >
        {model.title}
      </p>
    </button>
  );
}

export default ModelCard;
