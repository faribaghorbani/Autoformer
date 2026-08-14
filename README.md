# Adaptive Extensions to Autoformer for Long- Term Time Series Forecasting

This repository extends **Autoformer** for long-term time series forecasting on the **ETTm2** dataset.

## Proposed Modifications

- **Adaptive Multi-Scale Decomposition** — combines moving-average kernels of 13, 25, and 49 using learned weights.
- **Sample-Adaptive Lag Selection (2A)** — selects Top-$k$ Auto-Correlation delays independently for each sample.
- **Adaptive Period Routing (2B)** — uses a lightweight MLP to adaptively weight the selected periods. In our experiments, 2B is used together with 2A.

## Results

The combined **2A + 2B** model achieved the best overall performance, reducing average MSE from approximately **0.324 to 0.313** and MAE from **0.365 to 0.355** on ETTm2.

## Reference

Based on:

> Wu et al., _Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting_, NeurIPS 2021.

the source repository for Autoformer [github](https://github.com/thuml/Autoformer)

