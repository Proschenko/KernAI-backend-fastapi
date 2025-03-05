# tests/integration/test_labs.py
import pytest

@pytest.mark.asyncio
async def test_get_labs(client):
    response = client.get("/labs")
    assert response.status_code == 200
    assert isinstance(response.json(), list)  # Проверяем, что ответ — это список
    if response.json():
        assert "id" in response.json()[0]  # Проверяем наличие id
        assert "lab_name" in response.json()[0]  # Проверяем наличие lab_name
