import { defineStore } from "pinia"
import { ref, watch } from "vue"
import { useNetworkStore } from "@/stores/network"
import { useStocksStore } from "@/stores/stocks"
import { useCompanyStore } from "@/stores/company"
import { URLs } from "@/api/urls"
import { API } from "@/api"

export const useTrainingStore = defineStore("training", () => {
  const networkStore = useNetworkStore()
  const companyStore = useCompanyStore()
  const stocksStore = useStocksStore()

  const isDisabled = ref<boolean>(true)
  const isTraining = ref<boolean>(false)
  const isPaused = ref<boolean>(false)
  const epochCompleted = ref<number>(0)
  const mseHistory = ref<number[]>([])
  const mse = ref<number>(1.0)
  const sessionId = ref<string>("")
  const socket = ref<WebSocket | null>(null)

  // WS сообщения
  const onMessage = (event: MessageEvent) => {
    console.log(event)
    try {
      const data = JSON.parse(event.data)
      if (data.type === "training") {
        epochCompleted.value = data.epoch
        mseHistory.value = data.mse_history
        mse.value = data.loss
      }
      if (data.type === "training_completed") {
        stopTraining(event.data.predictions)
      }

    } catch (err) {
      console.error("Invalid WS message", event.data)
    }
  }

  const startTraining = async () => {
    const config = networkStore.config

    try {
      const response = await API.post(URLs.TRAINING.START, { data: config })
      sessionId.value = response.data.session_id

      socket.value = new WebSocket(`ws://localhost:8000/ws/${sessionId.value}`)

      socket.value.onopen = () => {
        console.log("WS Open")
        isTraining.value = true
        isPaused.value = false
        networkStore.setDisabled(true)
        companyStore.setDisabled(true)
        stocksStore.startLoading()
        mseHistory.value = []
      }

      socket.value.onmessage = onMessage

      socket.value.onclose = () => {
        console.log("WS Closed")
        isTraining.value = false
        isPaused.value = false
        networkStore.setDisabled(false)
        companyStore.setDisabled(false)
        stocksStore.stopLoading()
      }

      socket.value.onerror = (err) => {
        console.error("WS error", err)
      }

    } catch (err) {
      console.error("Training start failed", err)
    }
  }

  const cancelTraining = () => {
    if (socket.value && isTraining.value) {
      socket.value.send(JSON.stringify({ action: "stop" }))
      socket.value.close()
      socket.value = null
    }
    isTraining.value = false
    isPaused.value = false
    networkStore.setDisabled(false)
    companyStore.setDisabled(false)
    stocksStore.stopLoading()
  }

  const stopTraining = (predictions: any) => {
    if (socket.value && isTraining.value) {
      socket.value.close()
      socket.value = null
    }
    isTraining.value = false
    isPaused.value = false
    networkStore.setDisabled(false)
    companyStore.setDisabled(false)
    stocksStore.stopLoading()

    console.log(predictions)

  }

  const pauseTraining = () => {
    if (socket.value && isTraining.value && !isPaused.value) {
      socket.value.send(JSON.stringify({ action: "pause" }))
      isPaused.value = true
    }
  }

  const resumeTraining = () => {
    if (socket.value && isTraining.value && isPaused.value) {
      socket.value.send(JSON.stringify({ action: "resume" }))
      isPaused.value = false
    }
  }

  // // Watcher: если epochCompleted >= config.epochs => стоп
  // watch(epochCompleted, (val) => {
  //   if (networkStore.epochsCount && val >= networkStore.epochsCount) {
  //     stopTraining()
  //   }
  // })
  //
  // const getResults = async () => {
  //
  //   if (!isTraining) {
  //     return
  //   }
  //
  //   const url = URLs.TRAINING.RESULTS.replace("<sessionId>", sessionId.value)
  //   stocksStore.startLoading()
  //   API.get(url)
  //     .then((response) => {
  //       console.log(response)
  //     })
  // };

  return {
    isDisabled,
    setDisabled: (disabled: boolean) => (isDisabled.value = disabled),
    isTraining,
    isPaused,
    epochCompleted,
    mseHistory,
    mse,
    startTraining,
    stopTraining,
    cancelTraining,
    pauseTraining,
    resumeTraining,
    socket,
  }
})
