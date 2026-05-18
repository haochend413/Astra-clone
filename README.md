# Astra Running Results 

Hardware: A40 

## o4-mini results 

### Base vs Opt

| Kernel | LoC-Base | LoC-Opt. | ∆LoC  | Time-Base | Time-Opt. | Speedup |
| ------ | -------- | -------- | ----- | --------- | --------- | ------- |
| merge  | 124      | 246      | +100% | 61.96     | 52.11     | 1.20x   |
| rms    | 108      | 161      | +49%  | 158.9     | 194.1     | 0.84x   |
| silu   | 99       | 93       | -6%   | 28.13     | 18.76     | 1.50x   |

## Qwen Results

Qwen-3.5-9B and Qwen-Coder-Plus constantly fail to generate bug-free code within max turns. 

