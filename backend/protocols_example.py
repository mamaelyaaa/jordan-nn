from protocols_test import (
    CSVParser,
    TimeSeriesDatasetProtocol,
    TimeSeriesDataset,
    BatchGeneratorProtocol,
    RawData,
)

# Сырые данные из CSV
raw_data = CSVParser.load("data.csv")

# Создаем датасет
dataset: TimeSeriesDatasetProtocol = TimeSeriesDataset(
    raw_data=raw_data,
    feature_headers=["Open", "High", "Low", "Close", "Volume"],
    # Без сдвига по времени!
    targets_headers=["Close"],
)

# Создаем генератор батчей для временных рядов с окнами
batch_gen: BatchGeneratorProtocol = dataset.create_batch_generator(
    lookback=20,
    horizon=1,
    batch_size=32,
)

# Разделяем данные на обучающую и тестовую выборку
train_dataset, test_dataset = dataset.split(test_size=0.3)

for x_batch, y_batch in train_dataset.batch_generator:
    ...
