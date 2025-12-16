<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useCompanyStore } from '@/stores/company.ts'
import { storeToRefs } from 'pinia'

const store = useCompanyStore()

const { companies, selectedCompany, isDisabled } = storeToRefs(store)

const { chooseCompany } = store

onMounted(() => {
  store.fetchCompanies()
})
</script>

<template>
  <v-card title="Данные компаний">
    <v-form :disabled="isDisabled">
      <v-container>
        <v-row justify="stretch">
          <v-col cols="9">
            <v-select
              density="comfortable"
              v-model="selectedCompany"
              label="Название компании"
              :items="companies"
              item-title="name"
              item-value="symbol"
            />
          </v-col>

          <v-col cols="2">
            <v-btn color="primary" text="ОК" @click="chooseCompany" />
          </v-col>
        </v-row>
      </v-container>
    </v-form>
  </v-card>
</template>
