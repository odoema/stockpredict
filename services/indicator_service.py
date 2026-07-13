"""
Technical indicator service.

All indicators are implemented directly with pandas/numpy (no TA-Lib
dependency, which requires a compiled C library that isn't reliably
available on every host). Each function takes an OHLCV DataFrame and
returns a new DataFrame of indicator columns aligned to the same index.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Trend indicators
# ---------------------------------------------------------------------------

def sma(df: pd.DataFrame, window: int = 20) -> pd.Series:
    return df["Close"].rolling(window).mean().rename(f"SMA_{window}")


def ema(df: pd.DataFrame, window: int = 20) -> pd.Series:
    return df["Close"].ewm(span=window, adjust=False).mean().rename(f"EMA_{window}")


def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({"MACD": macd_line, "MACD_Signal": signal_line, "MACD_Hist": hist})


def adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = _true_range(df)
    atr_ = tr.ewm(alpha=1 / window, adjust=False).mean()

    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / window, adjust=False).mean() / atr_
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / window, adjust=False).mean() / atr_

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / window, adjust=False).mean().rename(f"ADX_{window}")


def parabolic_sar(df: pd.DataFrame, af_step: float = 0.02, af_max: float = 0.2) -> pd.Series:
    high, low = df["High"].values, df["Low"].values
    n = len(df)
    sar = np.zeros(n)
    if n == 0:
        return pd.Series(sar, index=df.index, name="PSAR")

    trend_up = True
    af = af_step
    ep = high[0]
    sar[0] = low[0]

    for i in range(1, n):
        prev_sar = sar[i - 1]
        if trend_up:
            sar[i] = prev_sar + af * (ep - prev_sar)
            sar[i] = min(sar[i], low[i - 1], low[max(i - 2, 0)])
            if high[i] > ep:
                ep = high[i]
                af = min(af + af_step, af_max)
            if low[i] < sar[i]:
                trend_up = False
                sar[i] = ep
                ep = low[i]
                af = af_step
        else:
            sar[i] = prev_sar + af * (ep - prev_sar)
            sar[i] = max(sar[i], high[i - 1], high[max(i - 2, 0)])
            if low[i] < ep:
                ep = low[i]
                af = min(af + af_step, af_max)
            if high[i] > sar[i]:
                trend_up = True
                sar[i] = ep
                ep = high[i]
                af = af_step

    return pd.Series(sar, index=df.index, name="PSAR")


# ---------------------------------------------------------------------------
# Momentum indicators
# ---------------------------------------------------------------------------

def rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).rename(f"RSI_{window}")


def williams_r(df: pd.DataFrame, window: int = 14) -> pd.Series:
    highest_high = df["High"].rolling(window).max()
    lowest_low = df["Low"].rolling(window).min()
    wr = -100 * (highest_high - df["Close"]) / (highest_high - lowest_low).replace(0, np.nan)
    return wr.rename(f"WilliamsR_{window}")


def cci(df: pd.DataFrame, window: int = 20) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    sma_tp = tp.rolling(window).mean()
    mad = tp.rolling(window).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return ((tp - sma_tp) / (0.015 * mad.replace(0, np.nan))).rename(f"CCI_{window}")


def roc(df: pd.DataFrame, window: int = 10) -> pd.Series:
    return (df["Close"].pct_change(window) * 100).rename(f"ROC_{window}")


def stochastic_oscillator(df: pd.DataFrame, k_window: int = 14, d_window: int = 3) -> pd.DataFrame:
    lowest_low = df["Low"].rolling(k_window).min()
    highest_high = df["High"].rolling(k_window).max()
    percent_k = 100 * (df["Close"] - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    percent_d = percent_k.rolling(d_window).mean()
    return pd.DataFrame({"Stoch_K": percent_k, "Stoch_D": percent_d})


# ---------------------------------------------------------------------------
# Volatility indicators
# ---------------------------------------------------------------------------

def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    tr = _true_range(df)
    return tr.ewm(alpha=1 / window, adjust=False).mean().rename(f"ATR_{window}")


def bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    mid = df["Close"].rolling(window).mean()
    std = df["Close"].rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return pd.DataFrame({"BB_Mid": mid, "BB_Upper": upper, "BB_Lower": lower})


def historical_volatility(df: pd.DataFrame, window: int = 21, trading_days: int = 252) -> pd.Series:
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    return (log_ret.rolling(window).std() * np.sqrt(trading_days) * 100).rename(f"HistVol_{window}")


def rolling_volatility(df: pd.DataFrame, window: int = 21) -> pd.Series:
    ret = df["Close"].pct_change()
    return (ret.rolling(window).std() * 100).rename(f"RollVol_{window}")


# ---------------------------------------------------------------------------
# Volume indicators
# ---------------------------------------------------------------------------

def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["Close"].diff()).fillna(0)
    return (direction * df["Volume"]).cumsum().rename("OBV")


def cmf(df: pd.DataFrame, window: int = 20) -> pd.Series:
    mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (df["High"] - df["Low"]).replace(0, np.nan)
    mfv = mfm * df["Volume"]
    return (mfv.rolling(window).sum() / df["Volume"].rolling(window).sum()).rename(f"CMF_{window}")


def mfi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    raw_money_flow = tp * df["Volume"]
    direction = np.sign(tp.diff()).fillna(0)
    positive_flow = raw_money_flow.where(direction > 0, 0).rolling(window).sum()
    negative_flow = raw_money_flow.where(direction < 0, 0).rolling(window).sum()
    money_ratio = positive_flow / negative_flow.replace(0, np.nan)
    return (100 - (100 / (1 + money_ratio))).rename(f"MFI_{window}")


def vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    return ((tp * df["Volume"]).cumsum() / df["Volume"].cumsum()).rename("VWAP")


def ease_of_movement(df: pd.DataFrame, window: int = 14) -> pd.Series:
    distance = ((df["High"] + df["Low"]) / 2).diff()
    box_ratio = (df["Volume"] / 1_000_000) / (df["High"] - df["Low"]).replace(0, np.nan)
    emv = distance / box_ratio
    return emv.rolling(window).mean().rename(f"EOM_{window}")


# ---------------------------------------------------------------------------
# Registry: maps a UI-facing key to (group, callable, output columns)
# ---------------------------------------------------------------------------

INDICATOR_REGISTRY = {
    # Trend
    "sma": {"group": "Trend", "label": "SMA (20)", "fn": lambda d: sma(d, 20).to_frame()},
    "ema": {"group": "Trend", "label": "EMA (20)", "fn": lambda d: ema(d, 20).to_frame()},
    "macd": {"group": "Trend", "label": "MACD", "fn": macd},
    "adx": {"group": "Trend", "label": "ADX (14)", "fn": lambda d: adx(d, 14).to_frame()},
    "psar": {"group": "Trend", "label": "Parabolic SAR", "fn": lambda d: parabolic_sar(d).to_frame()},
    # Momentum
    "rsi": {"group": "Momentum", "label": "RSI (14)", "fn": lambda d: rsi(d, 14).to_frame()},
    "williams_r": {"group": "Momentum", "label": "Williams %R (14)", "fn": lambda d: williams_r(d, 14).to_frame()},
    "cci": {"group": "Momentum", "label": "CCI (20)", "fn": lambda d: cci(d, 20).to_frame()},
    "roc": {"group": "Momentum", "label": "ROC (10)", "fn": lambda d: roc(d, 10).to_frame()},
    "stochastic": {"group": "Momentum", "label": "Stochastic Oscillator", "fn": stochastic_oscillator},
    # Volatility
    "atr": {"group": "Volatility", "label": "ATR (14)", "fn": lambda d: atr(d, 14).to_frame()},
    "bollinger": {"group": "Volatility", "label": "Bollinger Bands (20, 2)", "fn": bollinger_bands},
    "hist_vol": {"group": "Volatility", "label": "Historical Volatility (21d)", "fn": lambda d: historical_volatility(d, 21).to_frame()},
    "roll_vol": {"group": "Volatility", "label": "Rolling Volatility (21d)", "fn": lambda d: rolling_volatility(d, 21).to_frame()},
    # Volume
    "obv": {"group": "Volume", "label": "On-Balance Volume", "fn": lambda d: obv(d).to_frame()},
    "cmf": {"group": "Volume", "label": "Chaikin Money Flow (20)", "fn": lambda d: cmf(d, 20).to_frame()},
    "mfi": {"group": "Volume", "label": "Money Flow Index (14)", "fn": lambda d: mfi(d, 14).to_frame()},
    "vwap": {"group": "Volume", "label": "VWAP", "fn": lambda d: vwap(d).to_frame()},
    "eom": {"group": "Volume", "label": "Ease of Movement (14)", "fn": lambda d: ease_of_movement(d, 14).to_frame()},
}


def compute_indicators(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Compute the requested indicators and return them as columns aligned
    to `df`'s index. Unknown keys are silently skipped.
    """
    out = pd.DataFrame(index=df.index)
    for key in keys:
        spec = INDICATOR_REGISTRY.get(key)
        if spec is None:
            continue
        result = spec["fn"](df)
        out = out.join(result)
    return out


def list_indicators() -> dict:
    """Return indicators grouped for the UI's multi-select menus."""
    groups: dict[str, list[dict]] = {}
    for key, spec in INDICATOR_REGISTRY.items():
        groups.setdefault(spec["group"], []).append({"key": key, "label": spec["label"]})
    return groups
