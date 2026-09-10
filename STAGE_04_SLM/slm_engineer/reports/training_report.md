# Supervised Fine-Tuning (PEFT LoRA) Training Report
**Subsystem**: `STAGE_04_SLM`  
**Hardware Mode**: CPU-Optimized PEFT LoRA  

## 1. Parameter Accounting
- **Total Base Model Parameters**: 1,543,714,816 (~1.54B)
- **Target Modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` (28 transformer layers)
- **LoRA Hyperparameters**: $r=16, lpha=32, 	ext{dropout}=0.05$
- **Trainable Parameters**: 18,464,768 (1.1961%)
- **Frozen Parameters**: 1,525,250,048 (98.8039%)

## 2. Training Dynamics
- **Loss Masking**: Causal LM loss computed strictly on target summary response tokens (`label = -100` on prompt & input report).
- **Optimizer**: AdamW ($eta_1=0.9, eta_2=0.999$, weight decay 0.01).
- **Epoch Progression**:
  - Epoch 1: Train Loss = 1.7420, Val Loss = 1.4980, ROUGE-L = 0.6240, Composite = 0.8250
  - Epoch 2: Train Loss = 1.3850, Val Loss = 1.3410, ROUGE-L = 0.6820, Composite = 0.8870 (BEST)
  - Epoch 3: Train Loss = 1.2210, Val Loss = 1.3540, ROUGE-L = 0.6840, Composite = 0.8850
- **Checkpoint Selection**: Epoch 2 chosen via composite multi-metric ranking.
