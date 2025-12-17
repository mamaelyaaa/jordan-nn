export const API_BASE_URL = 'http://localhost:8000/api' as const

export const RESOURCES = {
  STOCKS: 'stocks',
  TRAINING: 'training',
} as const

export const URLs = {
  STOCKS: {
    /**
     * Список акций компаний.
     *
     * @allowed GET
     */
    LIST: `/${RESOURCES.STOCKS}/`,
    /**
     * Данные об акциях компании.
     * @desc Необходимо для построения первичного графика.
     *
     * @path symbol - Тикер компании
     * @allowed GET
     */
    HISTORY: `/${RESOURCES.STOCKS}/<symbol>/history/`,
  } as const,
  TRAINING: {
    /**
     * Старт обучения нейронной сети.
     * @desc Выводит уникальный id сессии, который нужно использовать для подключения к веб-сокету.
     *
     * @allowed POST
     */
    START: `/${RESOURCES.TRAINING}/start/`,
    /**
     * Результаты обучения с историческими данными.
     * @desc Доступны после остановки получения результатов от веб-сокетов
     *
     * @path  sessionId - Идентификатор сессии
     * @allowed GET
     */
    RESULTS: `/${RESOURCES.TRAINING}/<sessionId>/results/`,
    /**
     * Список всех сессий.
     *
     * @allowed GET
     */
    SESSIONS: `/${RESOURCES.TRAINING}/sessions/`,
  },
} as const
