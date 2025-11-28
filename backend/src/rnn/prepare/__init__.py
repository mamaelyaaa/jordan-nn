from typing import Literal

FeaturesType = Literal[
    "log_return",
    "pct_return",
    "high_rel",
    "low_rel",
    "close_rel",
    "candle_body",
    "rsi_14",
    "true_range_pct",
    "hv_14",
    "sma_14",
    "ema_14",
    "volatility_abs",
]

TargetType = Literal[
    "target_close_1d",
    "target_log_return_1d",
    "target_pct_return_1d",
    "target_volatility_1d",
    "target_direction_1d",
    "target_log_return_5d",
]
