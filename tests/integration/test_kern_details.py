# tests/integration/test_kern_details.py
import pytest

@pytest.mark.asyncio
async def test_get_kern_details(client):
    kern_id = "8665e63a-93dd-414c-9d46-0822f360faab"  # Укажи реальный UUID или фиктивный для теста
    response = client.get(f"/kern/{kern_id}")
    assert response.status_code == 200
    assert isinstance(response.json(), list)  # Ответ должен быть списком
    if response.json():
        assert "id" in response.json()[0]
        assert "kern_code" in response.json()[0]
 #добавить все поля TODO