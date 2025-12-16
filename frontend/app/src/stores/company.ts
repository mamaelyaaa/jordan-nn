import { defineStore } from 'pinia'
import {computed, ref} from "vue";
import {API} from "@/api";
import {URLs} from "@/api/urls.ts";
import {useErrorStore} from "@/stores/errors.ts";
import {useStocksStore} from "@/stores/stocks.ts";
import {useNetworkStore} from "@/stores/network.ts";
import {useTrainingStore} from "@/stores/training.ts";

export const useCompanyStore = defineStore("company", () => {

  const errorStore = useErrorStore();
  const networkStore = useNetworkStore();
  const trainingStore = useTrainingStore();
  const stocksStore = useStocksStore();

  const companies = ref<Array<{name: string, symbol: string}>>([])
  const selectedCompany = ref<string>("")
  const days = ref<number>(365)
  const isDisabled = ref<boolean>(false)

  const setSelectedCompany = (company: string) => {
    selectedCompany.value = company;
  };

  const setDays = (daysCount: number) => {
    days.value = daysCount;
  };

  const setDisabled = (disabled: boolean) => {
    isDisabled.value = disabled;
  }

  const setCompanies = (newCompanies: Array<{name: string, symbol: string}>) => {
    companies.value = newCompanies;
  }

  const fetchCompanies = async () => {
    API.get(URLs.STOCKS.LIST)
    .then((response) => {
      setCompanies(response.data.stocks);
    })
  };

  const chooseCompany = async () => {
    const symbol = selectedCompany.value;
    console.log(symbol)
    if (!symbol) {
      errorStore.setError("Компания не выбрана!");
      return
    }

    const url = URLs.STOCKS.HISTORY.replace("<symbol>", symbol)
    stocksStore.startLoading()
    API.get(url, { params: { days: days.value }})
      .then((response) => {
        stocksStore.updateStockHistory(response.data)
      })
      .then(() => {
        stocksStore.testPredicts = []
        stocksStore.trainPredicts = []
        trainingStore.mse = 1
        trainingStore.epochCompleted = 0
        trainingStore.mseHistory = []
        networkStore.setDisabled(false)
        trainingStore.setDisabled(false)
      })
  };

  return  {
    companies,
    selectedCompany,
    days,
    isDisabled,
    setDisabled,
    setSelectedCompany,
    setDays,
    setCompanies,
    fetchCompanies,
    chooseCompany
  }
})

