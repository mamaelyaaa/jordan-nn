<script setup lang="ts">
import { watch, ref } from 'vue'
import { useErrorStore } from '@/stores/errors'

const errorStore = useErrorStore()
const showSnackbar = ref(false)
const errorMessage = ref('')

watch(
  () => errorStore.error,
  (newError) => {
    if (newError) {
      errorMessage.value = newError
      showSnackbar.value = true
      errorStore.setError("")
    }
  }
)

</script>

<template>
  <v-snackbar
    v-model="showSnackbar"
    timeout="3000"
    vertical
    location="top"
  >
    <div class="text-subtitle-1 pb-2">Ошибка!</div>

    <p>{{errorMessage}}</p>

    <template v-slot:actions>
      <v-btn
        color="white"
        variant="text"
        @click="showSnackbar = false"
      >
        ЗАКРЫТЬ
      </v-btn>
    </template>
  </v-snackbar>
</template>

<style scoped>

</style>
