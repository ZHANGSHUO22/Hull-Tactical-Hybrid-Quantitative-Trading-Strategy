# Hull Tactical: Hybrid Quantitative Trading Strategy

This repository contains a robust quantitative trading strategy developed for the **Hull Tactical - Market Prediction** competition on Kaggle.

The solution implements a **hybrid ensemble approach** combining linear models (Ridge Regression), non-linear deep learning (Sharpe-Optimized MLP), and tree-based confidence gating (LightGBM). It features a sophisticated **Monte Carlo-based parameter optimization engine** that prioritizes tail-risk safety (CVaR/Expected Shortfall) over raw returns.


## Core Strategy & Architecture

The strategy is built upon a **Multi-Target Ensemble** framework where different models solve different parts of the trading puzzle:

### 1\. Model Ensemble

  * **Ridge Regression (Linear Baseline):** \* Target: *Excess Returns* (Denoised Alpha).
      * Role: Captures robust linear trends and mean-reversion signals.
  * **Sharpe-MLP (Deep Learning):**
      * Target: *Raw Returns*.
      * Loss Function: **Direct Sharpe Ratio Maximization** (Custom PyTorch Loss).
      * Role: Captures non-linear dependencies and optimizes for the specific risk/reward ratio.
  * **LightGBM (Confidence Gate):**
      * Target: *Directional Binary Classification*.
      * Role: Acts as a volatility filter. Signals are weighted by the model's confidence (`|p^2 - (1-p)^2|`).

### 2\. Robust Parameter Optimization (The "Secret Sauce")

Instead of simple grid search, this project uses a **Monte Carlo Robust Optimization** engine:

1.  **Walk-Forward OOF:** Generates out-of-sample predictions using a rolling time-series split.
2.  **Parallel Universes:** Simulates 1,000 alternative market history scenarios using bootstrap resampling.
3.  **CVaR Constraints:** Selects parameters ($w$ and $m$) that maximize the mean score while strictly constraining the **Expected Shortfall (ES 5%)** to prevent "Risk of Ruin."
4.  **Auto-Correction:** Automatically caps leverage if the tail risk exceeds safety thresholds.

### 3\. Dynamic Risk Management

  * **Volatility Targeting:** Position sizes are dynamically scaled inversely to realized volatility (`1/vol`).
  * **Leverage Hard-Cap:** Strict limits on maximum leverage to avoid the competition's volatility penalty.
  * **Regime Filtering:** Trend masks based on Moving Average crossovers (MA10 vs MA200).

## Project Pipeline

The code follows a strict pipeline to ensure reproducibility and prevent look-ahead bias:

1.  **Feature Engineering:** \* Lagged returns, Rolling Volatility/Skewness, Momentum Ratios, RSI, and Trend Masks.
      * Robust scaling to handle fat-tailed financial data.
2.  **Walk-Forward Validation:** \* Generates unbiased OOF (Out-Of-Fold) predictions for the ensemble.
3.  **Stress Testing:** \* Brute-force grid search combined with Monte Carlo simulation.
4.  **Inference:** \* Final prediction logic integrates `Ridge` + `NN` + `LGBM Confidence` with safety clips.

## Code Structure

```bash
├── main.py                  # Main script (Training, OOF, Optimization, Inference)
└── README.md                # Project documentation
```

## Usage

To run the simulation and training locally:

1.  **Install Dependencies:**

    ```bash
    pip install numpy pandas polars scikit-learn lightgbm torch tqdm
    ```

2.  **Run the script:**

    ```python
    python main.py
    ```

3.  **Inference Mode:**
    The script includes a `predict()` function designed for the Kaggle API. It handles state management (history buffer) and rolling feature engineering automatically.

## Key Code Snippets

**Custom Sharpe Loss (PyTorch):**

```python
class SharpeLoss(nn.Module):
    def forward(self, output, target_returns):
        strategy_returns = output * target_returns
        mean_ret = torch.mean(strategy_returns)
        std_ret = torch.std(strategy_returns) + 1e-6
        sharpe = (mean_ret / std_ret) * torch.sqrt(self.ann_factor)
        return -sharpe
```

**Robust Optimization Logic:**

```python
# Selects parameters that perform best in the worst 5% of simulated scenarios
if worst_05_mean < SAFETY_THRESHOLD:
    robust_metric = -99 # Reject risky parameters
else:
    robust_metric = mean_score
```

## ⚠️ Disclaimer

This project is for educational and research purposes only (specifically for the Kaggle Hull Tactical competition). It does not constitute financial advice. The strategy involves high leverage and carries significant risk.

-----

**Author:** [Zhang Shuo]
**Competition:** [Hull Tactical - Market Prediction](https://www.kaggle.com/competitions/hull-tactical-market-prediction)

-----
