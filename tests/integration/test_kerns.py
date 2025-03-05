# tests/integration/test_kerns.py
import pytest

@pytest.mark.asyncio
async def test_get_kerns(client):
    response = client.get("/kerns")
    assert response.status_code == 200
    assert isinstance(response.json(), list)  # Проверка, что ответ — список
    if response.json():
        assert "id" in response.json()[0]
        assert "kern_code" in response.json()[0]
