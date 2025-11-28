from pathlib import Path

from manager import RNNManager
from structure.activation import (
    TanhActivation,  # type: ignore
    # type: ignore
    # type: ignore
    # type: ignore
)
from structure.regularizer import (
    L2,  # type: ignore
    L1,  # type: ignore
)

STOCKS_DIR = Path(__file__).parent.parent.parent / "stocks"

if __name__ == "__main__":
    manager = RNNManager(
        features=[
            "ema_14",
            "rsi_14",
            "close_rel",
            "pct_return",
            # "low_rel",
            # "high_rel",
            # "volatility_abs",
        ],
        target=["target_close_1d"],
        hidden_neurons=128,
        hidden_activation=TanhActivation(),
        learning_rate=0.003,
        regularization=L2(lm=0.0003),
    )

    # Подготавливаем выборку
    stock = STOCKS_DIR / "apple.csv"
    raw = manager.load_and_prepare(source=stock, test_rate=0.3)

    # Обучаем модель
    mse = manager.train(epochs=1000)

    # Прогнозируем значения тестовой выборки (вместе с обучающей)
    predictions = manager.predict()

    # Строим графики
    manager.plot_predict_graphic(predictions=predictions)
