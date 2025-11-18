import numpy as np
import pandas as pd


class FeatureEngineering:

    def engine_features(self, df: pd.DataFrame) -> pd.DataFrame:
        result_df: pd.DataFrame = df.copy()

        # TODO Отрефакторить по логике (для меня заметочка: глянуть чат в ChatGPT)

        # Целевая переменная - цена закрытия на следующий день
        result_df["target_close"] = result_df["Close"].shift(-1)

        result_df["returns"] = result_df["Close"].pct_change(1)
        result_df["high_rel"] = result_df["High"] / result_df["Open"] - 1
        result_df["low_rel"] = result_df["Low"] / result_df["Open"] - 1
        result_df["close_rel"] = result_df["Close"] / result_df["Open"] - 1
        result_df["log_ret"] = np.log(result_df["Close"]).diff()

        true_range = np.maximum(
            df["High"] - df["Low"],
            np.maximum(
                abs(df["High"] - df["Close"].shift(1)),
                abs(df["Low"] - df["Close"].shift(1)),
            ),
        )
        result_df["true_range_pct%"] = true_range / result_df["Close"]

        result_df["volatility%"] = abs(result_df["Close"].pct_change())

        def rsi(series, period=14):
            delta = series.diff()
            gain = delta.clip(lower=0).ewm(alpha=1 / period).mean()
            loss = -delta.clip(upper=0).ewm(alpha=1 / period).mean()
            rs = gain / loss
            return 100 - 100 / (1 + rs)

        result_df["rsi"] = rsi(result_df["Close"])

        exp1 = result_df["Close"].ewm(span=12, adjust=False).mean()
        exp2 = result_df["Close"].ewm(span=26, adjust=False).mean()
        result_df["macd"] = exp1 - exp2
        result_df["macd_signal"] = result_df["macd"].ewm(span=9, adjust=False).mean()

        result_df = result_df.dropna()
        return result_df

    @staticmethod
    def add_minimal_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Создание минимального набора признаков
        """

        # Добавляем несколько эффективных признаков
        df["price_change_1d%"] = df["Close"].pct_change(1).shift(-1)
        df["daily_volatility"] = (df["High"] - df["Low"]) / df["Open"]
        return df

    @staticmethod
    def create_minimal_effective_features(df: pd.DataFrame) -> pd.DataFrame:
        """Минимальный, но очень эффективный набор"""

        # САМОЕ ВАЖНОЕ - 5 признаков
        # df["price_change_1d%"] = df["Close"].pct_change(1)
        # df["daily_volatility"] = (df["High"] - df["Low"]) / df["Open"]

        true_range = np.maximum(
            df["High"] - df["Low"],
            np.maximum(
                abs(df["High"] - df["Close"].shift(1)),
                abs(df["Low"] - df["Close"].shift(1)),
            ),
        )
        df["true_range_pct%"] = true_range / df["Close"]
        # df["volume_ratio"] = df["Volume"] / df["Volume"].rolling(2).mean()

        return df
