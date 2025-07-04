# app/redis_config.py
from celery import Celery
from redis import Redis

redis_client = Redis(host="redis", port=6379, db=0)

celery_app = Celery(
    "worker",
    broker="redis://redis:6379/0",
    backend="redis://redis:6379/0",
    include=["app.utils.celary.tasks"]  # Добавляем модуль с задачами
)
#TODO: Переделать для Linux
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    worker_pool = 'solo',
    # task_routes={
    #     'app.tasks.process_image_task': {'queue': 'celery'},
    # }
)
