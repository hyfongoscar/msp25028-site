import { Tooltip as RechartsTooltip } from 'recharts';

type Props = {
  active: boolean;
  payload: {
    name: string;
    value: number;
    dataKey: string;
    color: string;
  }[];
  label: string;
};

const Tooltip = ({ active, payload, label }: Props) => {
  if (active && payload && payload.length) {
    return (
      <RechartsTooltip
        content={
          <div
            style={{
              background: '#1A1F2C',
              border: '1px solid rgba(255,255,255,0.12)',
              borderRadius: 8,
              padding: '10px 14px',
              fontSize: 12,
              fontFamily: 'Inter, sans-serif',
            }}
          >
            <p style={{ color: '#8892a4', marginBottom: 6 }}>{label}</p>
            {payload.map((p: any) => (
              <p key={p.dataKey} style={{ color: p.color, margin: '2px 0' }}>
                {p.name}:{' '}
                <span style={{ color: '#f0f2f5', fontWeight: 600 }}>
                  ${p.value}
                </span>
              </p>
            ))}
          </div>
        }
      />
    );
  }
  return null;
};

export default Tooltip;
