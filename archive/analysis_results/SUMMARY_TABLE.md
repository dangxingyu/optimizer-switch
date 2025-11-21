# GSM8K 300M Finetuning - Summary

## Best Performance by Base Model and Optimizer

This table shows the **best accuracy** achieved across all learning rates for each combination.

| Base Model | Previous AdamW | Moonlight Muon |
|:-----------|:--------------:|:--------------:|
| **adamw_300m_8** | **5.08%**<br>(LR=1e-4) | **13.67%**<br>(LR=1e-3) |
| **muon_300m_8** | **12.11%**<br>(LR=1e-4) | **13.28%**<br>(LR=3e-4) |

## Key Findings

- **adamw_300m_8**: Moonlight Muon improves by **+8.59pp** over Previous AdamW
- **muon_300m_8**: Moonlight Muon improves by **+1.17pp** over Previous AdamW

**Overall**: Moonlight Muon achieves the best result of **13.67%** 🎉
