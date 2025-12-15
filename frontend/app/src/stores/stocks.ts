// stores/stocks.ts
import {defineStore} from 'pinia'
import {computed, ref} from 'vue'

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
  // State - храним данные акций
  const stockHistory = ref<StockHistory | null>(null)
  const isLoading = ref(false)

  // Геттер для проверки наличия данных
  const hasData = computed(() => {
    return stockHistory.value !== null &&
      stockHistory.value.data &&
      stockHistory.value.data.length > 0
  })

  // Геттер для преобразования данных в формат для свечного графика
  const candleStickSeries = computed(() => {
    if (!stockHistory.value?.data?.length) return []

    return [{
      name: stockHistory.value.symbol,
      data: stockHistory.value.data.map(item => ({
        x: new Date(item.Date),
        y: [item.Open, item.High, item.Low, item.Close]
      }))
    }]
  })

  const statistics = computed(() => {
    if (!stockHistory.value?.data?.length) return null

    const data = stockHistory.value.data

    if (data.length === 0) return null

    const latestIndex = data.length - 1
    const latest = data[latestIndex]
    const first = data[0]

    if (!latest || !first) return null

    const change = latest.Close - first.Open
    const changePercent = (change / first.Open) * 100

    return {
      symbol: stockHistory.value.symbol,
      days: stockHistory.value.days,
      currentPrice: latest.Close,
      openPrice: latest.Open,
      highPrice: latest.High,
      lowPrice: latest.Low,
      change,
      changePercent,
      isPositive: change >= 0,
      priceRange: latest.High - latest.Low,
      priceRangePercent: ((latest.High - latest.Low) / latest.Low) * 100
    }
  })

  const recentData = computed(() => {
    if (!stockHistory.value?.data?.length) return []

    const daysCount = Math.min(30, stockHistory.value.data.length)
    return stockHistory.value.data.slice(-daysCount)
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

    // Getters
    hasData,
    candleStickSeries,
    statistics,
    recentData,

    // Actions
    updateStockHistory,
    clearStockHistory,
    startLoading,
    stopLoading
  }
})
