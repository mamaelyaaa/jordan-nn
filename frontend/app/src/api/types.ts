type SuccessResponse<T> = {
  status: {
    type: "SUCCESS",
    code: number
  },
  data: T
}

type ErrorResponse = {
  status: {
    type: "ERROR",
    code: number
  }
  detail: string
}

export type ResponseHTTP<T> = SuccessResponse<T> | ErrorResponse

export interface CompaniesList {
  total: number;
  stocks: Array<{symbol: string, name: string}>;
}
