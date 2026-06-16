# SFT Configuration Reference

## configs/sft/sft-stage1.yaml

Initial SFT (5,000 steps)



## configs/sft/sft-stage2.yaml

Optimized SFT (10,000 steps)



## Key Differences from CPT

- Lower learning rate (1e-4 -> 5e-5)
- Response-only loss (mask user tokens)
- No packing (preserve conversation structure)
- load_best_model_at_end enabled
- Higher Islamic QA weight in stage 2
