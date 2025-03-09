import sys
import os
from pathlib import Path
# Добавляем корневую директорию проекта в sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from app.main import app  # Абсолютный импорт

@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client
        
from httpx import AsyncClient
from httpx import ASGITransport  # Используем ASGITransport для работы с FastAPI
@pytest_asyncio.fixture
async def async_client():
    """ Асинхронный клиент для тестов, если API использует async """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost:8000") as client:
        yield client