# Backend Fast API + PostgreSQL

1. Настройка среды и установка зависимостей

```
git clone "repo"
python -m venv .venv
pip install -r  requirements.txt
```

2. Настройка переменных окружения:

```
DATABASE_URL=postgresql+asyncpg://{user_db}:{password_db}@{url}:{port}/{db_name}
```

3. Поднятие redis для хранения очереди

```
docker run --name KernAI-redis-server -d -p 6379:6379 -v redis_data:/data redis
```

4. Запуск приложения:

```
uvicorn app.main:app --reload
```
