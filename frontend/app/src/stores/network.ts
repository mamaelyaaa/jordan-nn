// stores/network.ts
import { defineStore } from 'pinia'
import {ref, computed, watch, toRaw} from 'vue'
import {useCompanyStore} from "@/stores/company.ts";

export const useNetworkStore = defineStore('network', () => {

  const companyStore = useCompanyStore()

  // Доступность настроек
  const isDisabled = ref(true)
  const setDisabled = (disabled: boolean) => {
    isDisabled.value = disabled
  }

  // Эпохи
  const epochsCount = ref(1000)
  const setEpochsCount = (count: number) => {
    if (count >= 100 && count <= 10000) {
      epochsCount.value = count
    }
  }

  // Нейроны скрытого слоя
  const neuronsCount = ref(8)
  const setNeuronsCount = (count: number) => {
    if (count >= 1 && count <= 512) {
      neuronsCount.value = count
    }
  }

  // Функция активации скрытого слоя
  const activations = ref([
    {title: "Tanh", value: 'tanh'},
    {title: "Sigmoid", value: 'sigmoid'},
    {title: "Linear", value: 'linear'},
    {title: "ReLU", value: 'relu'}
  ])
  const selectedActivation = ref<string>("tanh")
  const setSelectedActivation = (activation: string) => {
    if (activation in activations) {
      selectedActivation.value = activation
    }
  }

  // Скорость обучения
  const learningRate = ref(0.001)
  const setLearningRate = (rate: number) => {
    if (rate >= 0.00001 && rate <= 1) {
      learningRate.value = rate
    }
  }

  // Признаки
  // const features = ref([
  //   {title: "Логарифм. доходность", value: "log_return"},
  //   {title: "Цена закрытия", value: "close"},
  //   {title: "Максимальная цена", value: "high"},
  //   {title: "Минимальная цена", value: "low"},
  //   {title: "Тело свечи", value: "candle_body"},
  //   {title: "RSI", value: "rsi_14"},
  //   {title: "EMA", value: "ema_14"},
  //   {title: "SMA", value: "sma_14"},
  //   {title: "Ист. волатильность", value: "hv_14"},
  //   {title: "Волатильность", value: "volatility"},
  // ])
  const features = ref([
    "log_return",
    "close",
    "high",
    "low",
    "candle_body",
    "rsi_14",
    "ema_14",
    "sma_14",
    "hv_14",
    "volatility",
  ])
  const selectedFeatures = ref<string[]>([])
  const setSelectedFeatures = (values: string[]) => {
    selectedFeatures.value = [...values] // Proxy уже не страшен
  }

  // Размер тестовой выборки
  const testRate = ref(0.3)
  const setTestRate = (rate: number) => {
    if (rate >= 0.01 && rate <= 0.99) {
      testRate.value = rate
    }
  }
  const testPercent = computed(() => {
    return Math.round(testRate.value * 100)
  })

  // Регуляризация
  const regularizations = ref(['L1', 'L2'])
  const selectedRegularization = ref<string>()
  const setSelectedRegularization = (regularization: string) => {
    selectedRegularization.value = regularization
  }

  // и скорость регуляризации
  const regularizationRate = ref(0.00000001)
  const setRegularizationRate = (rate: number) => {
    if (rate >= regularizationRateMin.value && rate <= regularizationRateMax.value) {
      regularizationRate.value = rate
    }
  }
  const regularizationRateMax = computed(() => {
    return learningRate.value * 0.99
  })
  const regularizationRateMin = computed(() => {
    return learningRate.value * 0.001
  })
  const isRegularizationDisabled = computed(() => {
    return !Boolean(selectedRegularization.value)
  })

  // Общая конфигурация для запроса
  const config = computed(() => ({
    stock_symbol: companyStore.selectedCompany,
    hidden_neurons: neuronsCount.value,
    epochs: epochsCount.value,
    learning_rate: learningRate.value,
    features: ["pct_return", ...selectedFeatures.value],
    test_rate: testRate.value,
    regularizer: selectedRegularization.value,
    regularizer_rate: regularizationRate.value,
    hidden_activation: selectedActivation.value,
    target: "close_1d",
  }))

  // Сброс настроек
  const resetToDefaults = () => {
    neuronsCount.value = 8
    epochsCount.value = 1000
    learningRate.value = 0.001
    selectedFeatures.value = []
    testRate.value = 0.5
    selectedRegularization.value = ''
    regularizationRate.value = 0.00000001
  }

  // Watchers
  watch(selectedRegularization, (newValue) => {
    if (!Boolean(newValue)) {
      regularizationRate.value = 0
    } else {
      regularizationRate.value = regularizationRateMax.value * 0.5
    }
  })

  watch(learningRate, () => {
    if (Boolean(selectedRegularization.value)) {
      regularizationRate.value = Math.min(
        regularizationRate.value,
        regularizationRateMax.value
      )
    }
  })

  return {
    // State
    neuronsCount,
    epochsCount,
    learningRate,
    selectedFeatures,
    testRate,
    selectedRegularization,
    regularizationRate,
    isDisabled,
    features,
    regularizations,
    activations,
    selectedActivation,

    // Computed
    testPercent,
    regularizationRateMax,
    regularizationRateMin,
    isRegularizationDisabled,
    config,

    // Actions
    setNeuronsCount,
    setEpochsCount,
    setLearningRate,
    setSelectedFeatures,
    setTestRate,
    setSelectedRegularization,
    setSelectedActivation,
    setRegularizationRate,
    setDisabled,
    resetToDefaults,

  }
})
