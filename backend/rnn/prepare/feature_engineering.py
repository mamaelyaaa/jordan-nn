import numpy as np
import pandas as pd


class FeatureEngineering:

    def engine_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Создаем копию чтобы не менять оригинал
        result_df = df.copy()

        # Целевая переменная - цена закрытия на следующий день
        result_df["target_close"] = result_df["Close"].shift(-1)

        # Добавляем фичи
        result_df = self.add_minimal_features(result_df)
        return result_df

    @staticmethod
    def add_minimal_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Создание минимального набора признаков
        """

        # Добавляем несколько эффективных признаков
        df["price_change_1d"] = df["Close"].pct_change(1)
        df["daily_volatility"] = (df["High"] - df["Low"]) / df["Open"]
        return df

    # @staticmethod
    # def create_minimal_effective_features(df: pd.DataFrame) -> pd.DataFrame:
    #     """Минимальный, но очень эффективный набор"""
    #
    #     # САМОЕ ВАЖНОЕ - 5 признаков
    #     df["price_change_1d"] = df["Close"].pct_change(1)
    #     df["daily_volatility"] = (df["High"] - df["Low"]) / df["Open"]
    #     df["close_position"] = (df["Close"] - df["Low"]) / (df["High"] - df["Low"])
    #     df["volume_ratio"] = df["Volume"] / df["Volume"].rolling(10).mean()
    #     df["price_vs_trend"] = df["Close"] / df["Close"].rolling(20).mean() - 1
    #
    #     # Лаги
    #     df["volatility_lag_1"] = df["daily_volatility"].shift(1)
    #     df["volume_lag_1"] = df["volume_ratio"].shift(1)
    #     return df
    #
    # @staticmethod
    # def add_volatility(df: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     Добавляет волатильность.
    #     Принимает значение от 0.1% до 10%:
    #
    #     < 0.5%: Низкая ликвидность, возможно резкое движение
    #     1.5-3%: Здоровая торговля
    #     >3%: Новостной драйвер, неопределенность
    #     >5%: Кризис/паника
    #
    #     :param df: Основной датафрейм
    #     :return: Новый датафрейм
    #     """
    #     true_range = np.maximum(
    #         df["High"] - df["Low"],
    #         np.maximum(
    #             abs(df["High"] - df["Close"].shift(1)),
    #             abs(df["Low"] - df["Close"].shift(1)),
    #         ),
    #     )
    #     df["true_range_pct"] = true_range / df["Close"]
    #     return df
    #
    # @staticmethod
    # def add_price_direction(df: pd.DataFrame) -> pd.DataFrame:
    #     df["price_direction"] = (df["Close"] > df["Close"].shift(1)).astype(int)
    #     df["strong_up_move"] = (df["Close"] > df["Close"].shift(1) * 1.01).astype(
    #         int
    #     )  # +1%
    #     df["strong_down_move"] = (df["Close"] < df["Close"].shift(1) * 0.99).astype(
    #         int
    #     )  # -1%
    #
    #     return df
    #
    # @staticmethod
    # def create_recommended_features(df: pd.DataFrame) -> pd.DataFrame:
    #     """Создание рекомендованного набора признаков"""
    #
    #     # 1. Базовые ценовые (обязательно)
    #     features_df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    #
    #     # 2. Ценовые изменения - САМЫЕ ВАЖНЫЕ!
    #     features_df["Close_pct_1d"] = df["Close"].pct_change(1)
    #     features_df["Close_pct_3d"] = df["Close"].pct_change(3)
    #     features_df["High_pct_1d"] = df["High"].pct_change(1)
    #     features_df["Low_pct_1d"] = df["Low"].pct_change(1)
    #     features_df["Open_pct_1d"] = df["Open"].pct_change(1)
    #
    #     # 3. Волатильность
    #     features_df["daily_range_pct"] = (df["High"] - df["Low"]) / df["Open"]
    #
    #     # True Range (учитывает гэпы)
    #     tr1 = df["High"] - df["Low"]
    #     tr2 = abs(df["High"] - df["Close"].shift(1))
    #     tr3 = abs(df["Low"] - df["Close"].shift(1))
    #     features_df["true_range_pct"] = (
    #         np.maximum(tr1, np.maximum(tr2, tr3)) / df["Close"]
    #     )
    #
    #     features_df["volatility_5d"] = df["Close"].pct_change().rolling(5).std()
    #
    #     # 4. Относительные уровни
    #     features_df["close_vs_high"] = (df["Close"] - df["Low"]) / (
    #         df["High"] - df["Low"]
    #     )
    #     features_df["distance_to_high"] = (df["High"] - df["Close"]) / df["Close"]
    #     features_df["distance_to_low"] = (df["Close"] - df["Low"]) / df["Close"]
    #
    #     # 5. Объем
    #     features_df["volume_pct_1d"] = df["Volume"].pct_change(1)
    #     features_df["volume_sma_10"] = df["Volume"].rolling(10).mean()
    #     features_df["volume_ratio_10d"] = df["Volume"] / features_df["volume_sma_10"]
    #
    #     # 6. Скользящие средние
    #     features_df["sma_5"] = df["Close"].rolling(5).mean()
    #     features_df["sma_20"] = df["Close"].rolling(20).mean()
    #     features_df["price_vs_sma_5"] = df["Close"] / features_df["sma_5"] - 1
    #     features_df["price_vs_sma_20"] = df["Close"] / features_df["sma_20"] - 1
    #     features_df["sma_5_vs_sma_20"] = (
    #         features_df["sma_5"] / features_df["sma_20"] - 1
    #     )
    #
    #     # 7. Временные
    #     if "Date" in df.columns:
    #         df["Date"] = pd.to_datetime(df["Date"])
    #         features_df["day_of_week"] = df["Date"].dt.dayofweek
    #         features_df["is_monday"] = (df["Date"].dt.dayofweek == 0).astype(int)
    #         features_df["is_friday"] = (df["Date"].dt.dayofweek == 4).astype(int)
    #
    #     # 8. Лаги (только самые важные)
    #     features_df["close_lag_1"] = df["Close"].shift(1)
    #     features_df["volume_lag_1"] = df["Volume"].shift(1)
    #     features_df["range_lag_1"] = features_df["daily_range_pct"].shift(1)
    #
    #     # Удаляем NaN
    #     features_df = features_df.dropna()
    #
    #     print(f"Создано {len(features_df.columns)} признаков")
    #     return features_df
