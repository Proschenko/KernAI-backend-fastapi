import sys
import os
from pathlib import Path

# Добавляем корневую директорию проекта в sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient
from app.main import app  # Абсолютный импорт

@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client
