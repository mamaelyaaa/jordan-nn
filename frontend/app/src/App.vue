<script setup lang="ts">

import {ref, computed, watch, onMounted} from 'vue'
import Company from "@/modules/Company.vue";
import Network from "@/modules/Network.vue";
import TrainingPanel from "@/modules/TrainingPanel.vue";
import Stocks from "@/modules/Stocks.vue"
import {useCompanyStore} from "@/stores/company.ts";
import Error from "@/modules/Error.vue";

const companyStore = useCompanyStore()

onMounted(async () => {
  await companyStore.fetchCompanies();
})

</script>

<template>
  <v-app>
    <v-main style="height: 100vh; padding: 3rem; background-color: #a3ccdf; display: flex;">
      <Error/>
      <v-container style="display: flex; width: 25vw; flex-direction: column; gap: 1rem;">
        <company/>
        <network/>
      </v-container>
      <v-container style="display: flex; width: 25vw; flex-direction: column; gap: 1rem; flex: 1">
        <stocks style="width: 100%"/>
<!--        <training-panel/>-->
      </v-container>
    </v-main>
  </v-app>
</template>

<style scoped> </style>
