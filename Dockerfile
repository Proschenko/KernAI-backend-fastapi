# Базовый образ Python
FROM python:3.11-slim

# Установим системные зависимости
RUN apt-get update && apt-get install -y gcc libpq-dev

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы зависимостей
COPY backend/requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь код проекта
COPY backend/ .

# Переменные окружения
ENV PYTHONUNBUFFERED=1

# Открываем порт uvicorn
EXPOSE 8000

# Команда по умолчанию - запуск uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
