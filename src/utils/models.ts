const MODELS = [
  {
    id: 'lstm',
    title: 'Model 1: Baseline LSTM',
    mse: '0.000231',
    accuracy: '64.7%',
    vizType: 'bar',
  },
  {
    id: 'qlstm',
    title: 'Model 2: QLSTM',
    mse: '0.000231',
    accuracy: '64.7%',
  },
  {
    id: 'custom_qnn',
    title: 'Model 3: CustomQNN',
    mse: '0.000358',
    accuracy: '60.2%',
  },
  {
    id: 'hybrid_qnn1',
    title: 'Model 4A: HybridQNN1',
    mse: '0.000182',
    accuracy: '68.9%',
  },
  {
    id: 'hybrid_qnn1_binary',
    title: 'Model 4B: HybridQNN1 (Binary output)',
    mse: '0.000182',
    accuracy: '68.9%',
  },
  {
    id: 'hybrid_qnn2',
    title: 'Model 5A: HybridQNN2',
    mse: '0.000294',
    accuracy: '63.5%',
  },
  {
    id: 'hybrid_qnn2_binary',
    title: 'Model 5B: HybridQNN2 (Binary output)',
    mse: '0.000294',
    accuracy: '63.5%',
  },
];

export default MODELS;
