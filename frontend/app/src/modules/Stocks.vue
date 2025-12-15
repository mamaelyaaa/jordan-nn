<script setup lang="ts">
import { computed, ref, onMounted, onUnmounted, watch } from 'vue'
import { useStocksStore } from '@/stores/stocks'
import { storeToRefs } from 'pinia'
import VueApexCharts from 'vue3-apexcharts'

const stocksStore = useStocksStore()
const { candleStickSeries, statistics, hasData, recentData, isLoading } = storeToRefs(stocksStore)

// Предсказания (mock / можно подключить к реальному стору с predictions)
const predictions = ref<number[]>([]) // сюда кладём массив предсказанных значений

// Реактивная высота для адаптивности
const chartContainer = ref<HTMLElement | null>(null)
const chartHeight = ref(600)

const updateChartHeight = () => {
  if (chartContainer.value) {
    const containerHeight = chartContainer.value.clientHeight
    chartHeight.value = containerHeight - 160
  }
}

onMounted(() => {
  updateChartHeight()
  window.addEventListener('resize', updateChartHeight)
})

onUnmounted(() => {
  window.removeEventListener('resize', updateChartHeight)
})

// Series с overlay линией predict
const combinedSeries = computed(() => {
  const series: any[] = []

  // Свечи
  if (candleStickSeries.value && candleStickSeries.value.length > 0) {
    series.push(...candleStickSeries.value)
  }

  // Линия predict
  if (predictions.value && predictions.value.length > 0) {
    const lineData = predictions.value.map((y, i) => {
      const x = candleStickSeries.value[0]?.data[i]?.x || new Date() // берём даты свечей
      return { x, y }
    })
    series.push({
      name: 'Predict',
      type: 'line' as const,
      data: lineData,
      color: '#002aff', // цвет линии
      stroke: { width: 2 },
    })
  }

  return series
})

const chartOptions = computed(() => ({
  chart: {
    height: chartHeight.value,
    type: 'candlestick' as const,
    stacked: false,
    toolbar: { show: true },
    background: 'transparent',
    foreColor: '#ffffff'
  },
  title: {
    text: statistics.value ? `${statistics.value.symbol} - Stock Price` : 'Stock Price',
    align: 'left' as const,
    style: { color: '#ffffff', fontSize: '18px', fontWeight: 'bold' }
  },
  grid: { borderColor: 'rgba(255,255,255,0.1)', strokeDashArray: 4 },
  xaxis: {
    type: 'datetime' as const,
    labels: { style: { colors: 'rgba(255,255,255,0.7)' } },
    axisBorder: { show: true, color: 'rgba(255,255,255,0.1)' },
    axisTicks: { show: true, color: 'rgba(255,255,255,0.1)' }
  },
  yaxis: {
    labels: { style: { colors: 'rgba(255,255,255,0.7)' }, formatter: (val: number) => `$${val.toFixed(2)}` },
    axisBorder: { show: true, color: 'rgba(255,255,255,0.1)' },
    title: { text: 'Price ($)', style: { color: 'rgba(255,255,255,0.7)' } }
  },
  plotOptions: {
    candlestick: {
      colors: { upward: '#00B746', downward: '#EF403C' },
      wick: { useFillColor: true }
    }
  },
  tooltip: { enabled: true, theme: 'dark', style: { fontSize: '12px' } }
}))
</script>

<template>
  <v-card style="flex: 1">
    <div ref="chartContainer" class="stocks-chart">
      <div v-if="isLoading" class="loading-state">
        <v-progress-circular indeterminate color="primary" size="64"></v-progress-circular>
      </div>

      <div v-else-if="hasData" class="candle-chart-container">
        <VueApexCharts
          type="candlestick"
          :height="chartHeight"
          :options="chartOptions"
          :series="combinedSeries"
        />
      </div>

      <div v-else class="no-data">
        <div class="no-data-title">Нет данных для отображения</div>
        <div class="no-data-text">
          Выберите компанию из списка слева и нажмите "ОК"
        </div>
      </div>
    </div>
  </v-card>
</template>

<style scoped> .stocks-chart { width: 100%; height: 100%; min-height: 600px; display: flex; flex-direction: column; background: transparent; } /* Состояние загрузки */ .loading-state { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 16px; padding: 40px; } .loading-text { font-size: 16px; color: rgba(0, 0, 0, 0.7); } /* График */ .candle-chart-container { flex: 1; min-height: 0; padding: 20px; } /* Сообщение при отсутствии данных */ .no-data { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 60px 40px; } .no-data-title { font-size: 22px; font-weight: 600; margin-bottom: 12px; text-align: center; } .no-data-text { font-size: 16px; text-align: center; max-width: 400px; line-height: 1.5; } </style>
