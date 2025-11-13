from network.jordan import JordanRNN
from network.training import HiddenLayer, OutputLayer, SigmoidActivation
from protocols_test import (
    CSVParser,
    TimeSeriesDatasetProtocol,
    TimeSeriesDataset,
    MinMaxScaler,
)
from protocols_test.batch_generator import TimeSeriesBatchGenerator

# Сырые данные из CSV
raw_data = CSVParser.load("data.csv")

# Создаем датасет
dataset: TimeSeriesDatasetProtocol = TimeSeriesDataset(
    raw_data=raw_data,
    feature_headers=["Open", "High", "Low", "Close", "Volume"],
    # Без сдвига по времени!
    targets_headers=["Close"],
    batch_generator=TimeSeriesBatchGenerator(
        lookback=20,
        horizon=1,
        batch_size=32,
    ),
    processor=MinMaxScaler(),
)

# Разделяем данные на обучающую и тестовую выборку
train_ds, test_ds = dataset.split(test_size=0.3)

# Обучаем нормализатор на обучающей выборке
train_ds.processor.fit(train_ds.features)

# Нормализуем все данные
train_ds.processor.normalize()
test_ds.processor.normalize()


# Создание модели
h_layer = HiddenLayer(activation=SigmoidActivation(), neurons=5)
o_layer = OutputLayer(activation=SigmoidActivation(), neurons=1)

model = JordanRNN(h_layer, o_layer, learning_rate=0.01)
model.train(training=..., targets=..., epochs=1000)
