# app/tasks.py
from .redis_config import celery_app
from .service import process_image

@celery_app.task(name='app.tasks.process_image_task')
def process_image_task(request_data: dict):
    return process_image(request_data)
