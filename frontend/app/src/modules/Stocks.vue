<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useStocksStore } from '@/stores/stocks'
import { useNetworkStore } from '@/stores/network'
import { storeToRefs } from 'pinia'
import VueApexCharts from 'vue3-apexcharts'
import {useTrainingStore} from "@/stores/training.ts";

/* ===== stores ===== */
const stocksStore = useStocksStore()
const networkStore = useNetworkStore()
const trainingStore = useTrainingStore()

const { candleStickSeries, hasData, isLoading } = storeToRefs(stocksStore)
const { testRate } = storeToRefs(networkStore)

/* ===== layout ===== */
const chartContainer = ref<HTMLElement | null>(null)
const chartHeight = ref(600)

const updateChartHeight = () => {
  if (chartContainer.value) {
    chartHeight.value = chartContainer.value.clientHeight - 160
  }
}

onMounted(() => {
  updateChartHeight()
  window.addEventListener('resize', updateChartHeight)
})

onUnmounted(() => {
  window.removeEventListener('resize', updateChartHeight)
})

/* ===== series ===== */
const series = computed(() => {
  if (!candleStickSeries.value?.length) return []

  const candles = candleStickSeries.value[0].data

  const trainLine = stocksStore.trainPredicts
    .map((y, i) => ({
      x: candles[i + 1]?.x,
      y,
    }))
    .filter((p) => p.x !== undefined)

  const testLine = stocksStore.testPredicts
    .map((y, i) => ({
      x: candles[i + trainLine.length + 1]?.x,
      y,
    }))
    .filter((p) => p.x !== undefined)

  return [
    {
      name: stocksStore.stockHistory?.symbol || 'Candles',
      type: 'candlestick',
      data: candles,
    },
    {
      name: 'Train Predict',
      type: 'line',
      data: trainLine,
    },
    {
      name: 'Test Predict',
      type: 'line',
      data: testLine,
    },
  ]
})

/* ===== конец train (КЛЮЧЕВО) ===== */
const trainEndX = computed<number | null>(() => {
  const data = candleStickSeries.value?.[0]?.data
  if (!data?.length) return null
  if (testRate.value <= 0 || testRate.value >= 1) return null

  const trainSize = Math.floor(data.length * testRate.value)
  const index = Math.min(trainSize - 1, data.length - 1)

  const x = data[index].x
  return x instanceof Date ? x.getTime() : x
})

/* ===== options (ТОЛЬКО computed) ===== */
const chartOptions = computed(() => ({
  chart: {
    type: 'candlestick',
    animations: { enabled: false },
    toolbar: { show: true },
    background: 'transparent',
    foreColor: '#ffffff',
  },

  grid: {
    borderColor: 'rgba(255,255,255,0.1)',
    strokeDashArray: 4,
  },

  xaxis: {
    type: 'datetime',
    labels: {
      style: { colors: 'rgba(255,255,255,0.7)' },
    },
  },

  yaxis: {
    labels: {
      style: { colors: 'rgba(255,255,255,0.7)' },
    },
  },

  colors: ['#00bcd4', '#2196f3', '#ff9800'],

  legend: {
    show: true,
    position: 'top',
    horizontalAlign: 'center',
    labels: {
      colors: '#000000',
      useSeriesColors: true,
    },
    markers: {
      width: 12,
      height: 12,
      radius: 6,
    },
  },

  annotations: {
    xaxis: stocksStore.separationDate
      ? [
          {
            x: stocksStore.separationDate,
            borderColor: '#FF9800',
            strokeDashArray: 4,
            label: {
              text: 'Train / Test',
              style: {
                background: '#FF9800',
                color: '#000',
              },
            },
          },
        ]
      : [],
  },
}))
</script>

<template>
  <v-card style="flex: 1">
    <div ref="chartContainer" class="stocks-chart">
      <div v-if="isLoading" class="loading-state">
        <v-progress-circular indeterminate size="64" />
        <p v-if="trainingStore.isTraining" class="loading-text">Обучение модели...</p>
        <p v-else class="loading-text">Загрузка данных...</p>
      </div>

      <div v-else-if="hasData" class="candle-chart-container">
        <VueApexCharts
          type="candlestick"
          :height="chartHeight"
          :options="chartOptions"
          :series="series"
        />
      </div>

      <div v-else class="no-data">
        <p class="no-data-title">Нет данных</p>
        <p class="no-data-text">Выберите компанию и нажмите ОК</p>
      </div>
    </div>
  </v-card>
</template>

<style scoped>
.stocks-chart {
  width: 100%;
  height: 100%;
  min-height: 600px;
  display: flex;
  flex-direction: column;
  background: transparent;
} /* Состояние загрузки */
.loading-state {
  flex: 1;
  display: flex;
  color: rgb(0, 42, 255);
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 16px;
  padding: 40px;
}
.loading-text {
  font-size: 18px;
  color: rgb(0, 42, 255);
} /* График */
.candle-chart-container {
  flex: 1;
  min-height: 0;
  padding: 20px;
} /* Сообщение при отсутствии данных */
.no-data {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 60px 40px;
}
.no-data-title {
  font-size: 22px;
  font-weight: 600;
  margin-bottom: 12px;
  text-align: center;
}
.no-data-text {
  font-size: 16px;
  text-align: center;
  max-width: 400px;
  line-height: 1.5;
}
</style>
