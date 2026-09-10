# STAGE 2: DEEP LEARNING ARCHITECTURAL SPECIFICATION REPORT

## 1. Sub-Network Architectures
1. **Visual CNN**: EfficientNet-B0 pretrained feature extractor ($d_v = 128$).
2. **Temporal LSTM**: 1-layer LSTM with sequence packing & masking ($d_t = 64$).
3. **Clinical MLP**: 2-layer Dense network with BatchNorm, ReLU, and Dropout ($d_c = 64$).

## 2. Regularization & Overfitting Controls
- **Strict Patient-Level Splitting**: Zero patient contamination across Train, Validation, and Test.
- **Dropout**: 0.3 across all sub-networks and fusion head.
- **Weight Decay**: L2 penalty (1e-3 for CNN, 1e-4 for MLP/LSTM) in AdamW optimizer.
- **Batch Normalization**: Stabilizes training dynamics across multi-scale feature spaces.
- **Early Stopping**: Monitored on Validation Macro F1.
