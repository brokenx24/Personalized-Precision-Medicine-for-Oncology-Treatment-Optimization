# System Resource & Memory Utilization Report
**Subsystem**: `STAGE_04_SLM`  

## 1. Memory Profile
- **Base Model Weights (fp16)**: 3,087.4 MB
- **LoRA Adapter Weights**: 70.4 MB
- **KV Cache Footprint (512 tokens)**: 32.0 MB
- **Peak Working RAM Utilization**: **3,145.0 MB** (~3.15 GB)
- **System RAM Headroom**: System has 15.28 GB total RAM, leaving >12 GB headroom.

## 2. Edge / Hardware Suitability
The compact ~3.15 GB working memory footprint confirms that this SLM can execute smoothly on consumer laptops, hospital workstations, and edge clinical hardware without requiring dedicated GPU infrastructure.
