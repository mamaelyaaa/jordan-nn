DC = docker compose
EXEC = docker exec -it
LOGS = docker logs
APP_FILE = docker-compose.yml
APP_CONTAINER = jordan-backend

.PHONY: app
app:
	${DC} -f ${APP_FILE} up --build -d

.PHONY: app-logs
app-logs:
	${LOGS} ${APP_CONTAINER} -f

.PHONY: app-down
app-down:
	${DC} -f ${APP_FILE} down

.PHONY: app-shell
app-shell:
	${EXEC} ${APP_CONTAINER} bash