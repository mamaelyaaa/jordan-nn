<script setup lang="ts">
import { useTrainingStore } from "@/stores/training"
import { storeToRefs } from "pinia"
import { ref, watch } from "vue"
import VueApexCharts from "vue3-apexcharts"

// иконки
import { mdiPlay, mdiStop, mdiPause } from '@mdi/js'
import type { ApexOptions } from "apexcharts"

const store = useTrainingStore()
const { isDisabled, isTraining, isPaused, epochCompleted, mseHistory, mse } = storeToRefs(store)
const { startTraining, stopTraining, cancelTraining, pauseTraining, resumeTraining } = store

function onPauseClick() {
  if (isPaused.value) resumeTraining()
  else pauseTraining()
}

// Отображаем только последние 100 точек
const MAX_POINTS = 100

// Ref на компонент графика
const apexChartRef = ref<InstanceType<typeof VueApexCharts> | null>(null)

// Опции ApexChart
const chartOptions: ApexOptions = {
  chart: {
    id: 'mse-chart',
    animations: { enabled: false },
    toolbar: { show: false },
    sparkline: { enabled: false }
  },
  xaxis: {
    categories: [] as number[],
    labels: { show: false },
    axisBorder: { show: false },
    axisTicks: { show: false },
  },
  yaxis: {
    min: 0,
    max: 1,
    title: { text: '' },
    labels: {
      show: true,
      formatter: (val) => val.toFixed(2) // два знака после запятой
    },
    axisBorder: { show: true },
    axisTicks: { show: true },
  },
  stroke: { curve: 'smooth' },
  tooltip: { enabled: false },
  legend: { show: false },
  title: { text: '' },
  grid: {
    show: true,
    borderColor: '#e0e0e0',
    row: { colors: undefined },
    column: { colors: undefined },
    yaxis: { lines: { show: true } },
    xaxis: { lines: { show: false } }
  }
}



// Изначально пустая серия
const chartSeries = ref([{ name: 'MSE', data: [] }])

// Обновляем график при изменении mseHistory, только последние MAX_POINTS
watch(mseHistory, (newVal) => {
  if (!apexChartRef.value || !newVal) return

  const data = newVal.slice(-MAX_POINTS)
  chartSeries.value = [{ name: 'MSE', data }]

  // Обновляем график через ApexCharts метод updateSeries
  apexChartRef.value.updateSeries([{ name: 'MSE', data }])
})
</script>

<template>
  <v-card
    :disabled="isDisabled"
    :title="`Эпохи: ${epochCompleted}, MSE: ${mse}`"
    style="display: flex; flex-direction: column; height: 225px"
  >
    <v-card-text style="display: flex; flex: 1; gap: 16px; padding: 0; align-items: center;">
      <!-- Колонка кнопок -->
      <v-container
        class="d-flex flex-column"
        style="gap: 16px; width: 64px; padding: 16px 0 16px 16px;"
      >
        <!-- PLAY / STOP -->
        <v-tooltip location="right">
          <template #activator="{ props }">
            <v-btn
              v-bind="props"
              icon
              size="48"
              color="primary"
              @click="isTraining ? cancelTraining() : startTraining()"
            >
              <v-icon :icon="isTraining ? mdiStop : mdiPlay" />
            </v-btn>
          </template>
          <span>{{ isTraining ? 'Отмена' : 'Начать обучение' }}</span>
        </v-tooltip>

        <!-- PAUSE / RESUME -->
        <v-tooltip location="right">
          <template #activator="{ props }">
          <span v-bind="props">
            <v-btn
              icon
              size="48"
              color="secondary"
              :disabled="!isTraining"
              @click="onPauseClick"
            >
              <v-icon :icon="mdiPause" />
            </v-btn>
          </span>
          </template>
          <span>{{ isPaused ? 'Продолжить' : 'Пауза' }}</span>
        </v-tooltip>
      </v-container>

      <!-- График занимает всё оставшееся пространство -->
      <div style="flex: 1; padding: 16px">
        <vue-apex-charts
          ref="apexChartRef"
          v-if="mseHistory && mseHistory.length"
          type="line"
          height="125"
          width="100%"
          :options="chartOptions"
          :series="chartSeries"
        />
      </div>
    </v-card-text>
  </v-card>
</template>
