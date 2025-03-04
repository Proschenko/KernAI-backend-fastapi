#Запуск приложения:

```
uvicorn app.main:app --reload
```

#Запуск Celary
```
celery -A app.redis_config.celery_app worker --loglevel=info
```