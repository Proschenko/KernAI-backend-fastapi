# Базовый образ Python
FROM python:3.11-slim

# Установим системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 gcc libpq-dev

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы зависимостей
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь код проекта
COPY app/ .

WORKDIR /

# Открываем порт uvicorn
EXPOSE 8000

# Команда по умолчанию - запуск uvicorn
#CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
#CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload","--ssl-keyfile","/ssl/tls.key","--ssl-certfile","/ssl/tls.pem"]
#CMD uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --ssl-keyfile /ssl/tls.key --ssl-certfile /ssl/tls.pem; celery -A app.utils.celary.redis_config.celery_app worker --loglevel=INFO --logfile=/celery.log
CMD uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --ssl-keyfile /ssl/tls.key --ssl-certfile /ssl/tls.pem
