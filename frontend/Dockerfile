FROM node:20-bookworm-slim

WORKDIR /app

# Копируем файлы зависимостей
COPY app/package.json app/package-lock.json ./

# Устанавливаем ВСЕ зависимости
RUN npm install --no-audit --legacy-peer-deps

# Копируем остальные файлы
COPY . .

# Экспонируем порт
EXPOSE 5173

# Запускаем сервер разработки
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"]