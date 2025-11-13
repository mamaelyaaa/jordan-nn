from protocols import (
    CSVParser,
    DatasetProtocol,
    TimeSeriesDataset,
    TimeSeriesBatchGenerator,
    BatchGeneratorProtocol,
)

# Сырые данные из CSV
raw_data = CSVParser.load("data.csv")

# Создаем датасет
dataset: DatasetProtocol = TimeSeriesDataset(raw_data)

# Создаем генератор батчей для временных рядов с окнами
dataset.create_batch_generator(
    lookback=20,
    horizon=1,
    batch_size=32,
)

# Разделяем данные на обучающую и тестовую выборку
train_dataset, test_dataset = dataset.split(test_size=0.3)

# Создаем генератор батчей для временных рядов с окнами
train_gen: BatchGeneratorProtocol = train_dataset.batch_generator
test_gen: BatchGeneratorProtocol = test_dataset.batch_generator

# TODO Реализовать iter для быстрой итерации по батчам генератора
for _ in train_gen:
    pass
