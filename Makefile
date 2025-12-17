DC = docker compose
EXEC = docker exec -it
LOGS = docker logs
APP_FILE = docker-compose.yml
BACKEND_CONTAINER = jordan-backend
FRONTEND_CONTAINER = jordan-frontend


.PHONY: all
all:
	${DC} -f ${APP_FILE} up --build -d

.PHONY: backend
backend:
	${DC} -f ${APP_FILE} up --build -d backend

.PHONY: frontend
frontend:
	${DC} -f ${APP_FILE} up --build -d frontend

.PHONY: logs
logs:
	${DC} -f ${APP_FILE} logs -f

.PHONY: down
down:
	${DC} -f ${APP_FILE} down

.PHONY: down-v
down-v:
	${DC} -f ${APP_FILE} down -v

.PHONY: backend-shell
backend-shell:
	${EXEC} ${BACKEND_CONTAINER} bash

.PHONY: frontend-shell
frontend-shell:
	${EXEC} ${FRONTEND_CONTAINER} bash

.PHONY: rebuild
rebuild:
	${DC} -f ${APP_FILE} down
	${DC} -f ${APP_FILE} up --build -d

.PHONY: stop
stop:
	${DC} -f ${APP_FILE} stop

.PHONY: start
start:
	${DC} -f ${APP_FILE} start