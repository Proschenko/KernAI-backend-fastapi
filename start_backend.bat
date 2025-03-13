@echo off
set VENV_PATH=.venv\Scripts\activate

start "FastAPI Server" cmd /k "call %VENV_PATH% && uvicorn app.main:app --reload"
start "Celery Worker" cmd /k "call %VENV_PATH% && celery -A app.utils.celary.redis_config.celery_app worker --loglevel=info"
