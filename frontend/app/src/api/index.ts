import axios from 'axios'
import { API_BASE_URL } from '@/api/urls.ts'
import { useErrorStore } from '@/stores/errors.ts'

export const API = axios.create({
  baseURL: API_BASE_URL,
  timeout: 5000,
  headers: {
    'Content-Type': 'application/json',
  },
})

API.interceptors.response.use(
  (response) => {
    return response
  },
  (error) => {
    let errorStore
    try {
      errorStore = useErrorStore()
    } catch (e) {
      console.error('Error store not available:', e)
      return Promise.reject(error)
    }

    console.log(error)

    const errorMessage = error.response?.data?.detail || 'Неизвестная ошибка'

    errorStore.setError(errorMessage)

    return Promise.reject(error)
  },
)
