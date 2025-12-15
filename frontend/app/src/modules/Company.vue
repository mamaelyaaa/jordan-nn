<script setup lang="ts">
import {computed, onMounted, ref} from 'vue'
import {useCompanyStore} from "@/stores/company.ts";
import {storeToRefs} from "pinia";

const store = useCompanyStore()

const {companies, selectedCompany, days, isDisabled} = storeToRefs(store)

const { chooseCompany } = store

onMounted(() => {
  store.fetchCompanies()
})

</script>

<template>
  <v-card
    title="Данные компаний"
    :disabled="isDisabled"
  >
    <v-form style="margin: 15px 30px 30px 30px;">
      <v-row>
        <v-select
          v-model="selectedCompany"
          label="Название компании"
          :items="companies"
          item-title="name"
          item-value="symbol"
          block
        />
      </v-row>
      <v-row>
        <v-slider
          v-model="days"
          label="Диапазон дней"
          :step="1"
          :min="1"
          :max="1095"
          block
        >
          <template v-slot:append>
            {{ days }}
          </template>
        </v-slider>
      </v-row>
      <v-row>
        <v-btn
          text="ОК"
          block
          @click="chooseCompany"
        />
      </v-row>
    </v-form>
  </v-card>
</template>
