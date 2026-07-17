const MODELS = [
  {
    id: 'lstm',
    title: 'Model 1: Baseline LSTM',
    rmse: '0.0164',
    mae: '0.0162',
    r_squared: '0.09854',
  },
  {
    id: 'qlstm',
    title: 'Model 2: QLSTM',
    rmse: '0.0111',
    mae: '0.0089',
    r_squared: '0.9933',
  },
  {
    id: 'custom_qnn',
    title: 'Model 3: CustomQNN',
    rmse: '4.7222',
    mae: '2.5267',
  },
  {
    id: 'hybrid_qnn1',
    title: 'Model 4A: HybridQNN1',
    rmse: '50.0699',
    mae: '40.8827',
    r_squared: '-1.9048',
  },
  {
    id: 'hybrid_qnn1_binary',
    title: 'Model 4B: HybridQNN1 (Binary)',
    accuracy: '55.1%',
    roc_auc: '0.5633',
  },
  {
    id: 'hybrid_qnn2',
    title: 'Model 5A: HybridQNN2',
    rmse: '21.6792',
    mae: '16.2030',
    r_squared: '0.4554',
  },
  {
    id: 'hybrid_qnn2_binary',
    title: 'Model 5B: HybridQNN2 (Binary)',
    accuracy: '55.1%',
    roc_auc: '0.5638',
  },
];

export default MODELS;
