// stores/stocks.ts
import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import { useNetworkStore } from '@/stores/network.ts'

export interface StockDataPoint {
  Date: string
  Open: number
  High: number
  Low: number
  Close: number
}

export interface StockHistory {
  symbol: string
  days: number
  data: StockDataPoint[]
}

export const useStocksStore = defineStore('stocks', () => {
  const networkStore = useNetworkStore()

  const stockHistory = ref<StockHistory | null>(null)

  const trainPredicts = ref<number[]>([])
  const testPredicts = ref<number[]>([])
  const separationDate = computed(() => {
    if (!stockHistory.value?.data?.length) return null

    const data = stockHistory.value.data
    const total = data.length

    const testRate = networkStore.testRate
    if (testRate <= 0 || testRate >= 1) return null

    const trainSize = Math.floor(total * (1 - testRate))
    const lastTrainIndex = trainSize - 1
    if (lastTrainIndex < 0) return null
    console.log(new Date(data[lastTrainIndex].Date).getTime())
    return new Date(data[lastTrainIndex].Date).getTime()
  })

  const isLoading = ref(false)

  // Геттер для проверки наличия данных
  const hasData = computed(() => {
    return (
      stockHistory.value !== null && stockHistory.value.data && stockHistory.value.data.length > 0
    )
  })

  const candleStickSeries = computed(() => {
    if (!stockHistory.value?.data?.length) return []

    return [
      {
        name: stockHistory.value.symbol,
        data: stockHistory.value.data.map((item) => ({
          x: new Date(item.Date),
          y: [item.Open, item.High, item.Low, item.Close],
        })),
      },
    ]
  })

  // Действие для обновления данных с состоянием загрузки
  const updateStockHistory = async (dataPromise: Promise<StockHistory> | StockHistory) => {
    isLoading.value = true
    try {
      if (dataPromise instanceof Promise) {
        stockHistory.value = await dataPromise
      } else {
        stockHistory.value = dataPromise
      }
    } catch (error) {
      console.error('Error updating stock history:', error)
      throw error
    } finally {
      isLoading.value = false
    }
  }

  // Действие для очистки данных
  const clearStockHistory = () => {
    stockHistory.value = null
    isLoading.value = false
  }

  // Действие для начала загрузки
  const startLoading = () => {
    isLoading.value = true
  }

  // Действие для остановки загрузки
  const stopLoading = () => {
    isLoading.value = false
  }

  return {
    // State
    stockHistory,
    isLoading,

    trainPredicts,
    testPredicts,
    separationDate,

    // Getters
    hasData,
    candleStickSeries,

    // Actions
    updateStockHistory,
    clearStockHistory,
    startLoading,
    stopLoading,
  }
})
