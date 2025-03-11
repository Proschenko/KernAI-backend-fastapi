# Backend Fast API + PostgreSQL

1. Настройка среды и установка зависимостей

```
git clone "repo"
python -m venv .venv
\.venv\Scripts\activate
pip install -r  requirements.txt
```

2. Настройка переменных окружения:

```
DATABASE_URL=postgresql+asyncpg://{user_db}:{password_db}@{url}:{port}/{db_name}

KEYCLOAK_SERVER_URL=https://your-keycloak-server/auth
KEYCLOAK_REALM=your-realm
KEYCLOAK_CLIENT_ID=your-client-id
KEYCLOAK_CLIENT_SECRET=your-client-secret
```

3. Поднятие redis для хранения очереди

```
docker run --name KernAI-redis-server -d -p 6379:6379 -v redis_data:/data redis
```

4. Запуск Celary в отдельной консоли

```
celery -A app.redis_config.celery_app worker --loglevel=info
```

5. Запуск приложения в отдельной консоли

```
uvicorn app.main:app --reload
```
