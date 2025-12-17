import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useErrorStore = defineStore('error', () => {
  const error = ref<string>('')

  const setError = (err: string) => {
    error.value = err
  }

  return {
    error,
    setError,
  }
})
