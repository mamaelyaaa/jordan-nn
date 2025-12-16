<!-- components/Network.vue -->
<script setup lang="ts">
import { useNetworkStore } from '@/stores/network'
import { storeToRefs } from 'pinia'

const networkStore = useNetworkStore()

const {
  neuronsCount,
  epochsCount,
  learningRate,
  selectedFeatures,
  testRate,
  selectedRegularization,
  selectedActivation,
  activations,
  regularizationRate,
  features,
  regularizations,
  testPercent,
  regularizationRateMax,
  regularizationRateMin,
  isRegularizationDisabled,
  isDisabled,
} = storeToRefs(networkStore)
</script>

<template>
  <v-card
    :disabled="isDisabled"
    title="Конфигурация сети"
    style="
    flex: 1;
    display: flex;
    flex-direction: column;
    min-height: 0;
  "
  >
    <v-card-text
      style="
    flex: 1;
    overflow-y: auto;
    min-height: 0;
  "
    >
      <v-form style="margin: 15px 30px 30px 30px;">
        <v-row style="gap: 15px;">
          <v-number-input
            label="Кол-во эпох"
            control-variant="hidden"
            density="compact"
            v-model="epochsCount"
            :max="10000"
            :min="100"
            :step="100"
            @update:model-value="networkStore.setEpochsCount"
          ></v-number-input>

          <v-number-input
            label="Кол-во нейронов"
            density="compact"
            control-variant="hidden"
            v-model="neuronsCount"
            :max="512"
            :min="1"
            :step="8"
            @update:model-value="networkStore.setNeuronsCount"
          ></v-number-input>
        </v-row>

        <v-row>
          <v-select
            v-model="selectedActivation"
            density="compact"
            label="Функция активации"
            :items="activations"
            item-title="title"
            item-value="value"
            block
            @update:model-value="networkStore.setSelectedActivation"
          />
        </v-row>

        <v-row>
          <v-number-input
            label="Скорость обучения"
            density="compact"
            control-variant="hidden"
            v-model="learningRate"
            :max="1"
            :precision="5"
            :min="0.00001"
            :step="0.001"
            @update:model-value="networkStore.setLearningRate"
          />
        </v-row>

        <v-row>
          <v-combobox
            v-model="selectedFeatures"
            :items="features"
            item-title="title"
            item-value="value"
            multiple
            chips
          />
        </v-row>

        <v-row>
          <v-slider
            v-model="testRate"
            label="Размер тестовой выборки"
            :step="0.01"
            :min="0.01"
            :max="0.99"
            block
            @update:model-value="networkStore.setTestRate"
          >
            <template v-slot:append>
              {{ testPercent }}%
            </template>
          </v-slider>
        </v-row>

        <v-row style="gap: 1rem;">
          <v-select
            density="compact"
            v-model="selectedRegularization"
            label="Регуляризация"
            :items="regularizations"
            block
            clearable
            @update:model-value="networkStore.setSelectedRegularization"
          />

          <v-number-input
            density="compact"
            v-if="!isRegularizationDisabled"
            label="Скорость"
            control-variant="hidden"
            v-model="regularizationRate"
            :max="regularizationRateMax"
            :precision="8"
            :min="regularizationRateMin"
            @update:model-value="networkStore.setRegularizationRate"
          />
        </v-row>
      </v-form>
    </v-card-text>
  </v-card>
</template>

<style scoped>

</style>
