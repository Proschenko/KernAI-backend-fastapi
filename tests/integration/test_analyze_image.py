# tests/integration/test_analyze_image.py
import pytest
from uuid import uuid4
from datetime import datetime

@pytest.mark.asyncio
async def test_analyze_image(client):
    request_data = {
        "username": "test_user",
        "image_path": "path/to/test_image.jpg",
        "codes": ["K-123", "K-456"],
        "lab_id": "uuid4()  # Пример UUID для теста"
    }
    response = client.post("/analyze_img", json=request_data)
    assert response.status_code == 200
    assert "results" in response.json()  # Должен быть результат обработки
    assert isinstance(response.json()["results"], list)  # Результаты должны быть в виде списка
