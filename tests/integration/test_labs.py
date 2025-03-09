import pytest
import httpx

API_URL = "http://localhost:8000/labs"

@pytest.mark.asyncio
async def test_get_labs_structure():
    async with httpx.AsyncClient() as client:
        response = await client.get(API_URL)
    
    # Проверяем, что статус-код 200
    assert response.status_code == 200, f"Expected status 200, got {response.status_code}"
    
    # Проверяем, что JSON-ответ это список
    json_response = response.json()
    assert isinstance(json_response, list), "Response is not a list"
    
    # Проверяем, что каждый элемент в списке это словарь с нужными ключами
    for item in json_response:
        assert isinstance(item, dict), "Item is not a dictionary"
        assert "id" in item, "Missing 'id' in item"
        assert "lab_name" in item, "Missing 'lab_name' in item"
        assert isinstance(item["id"], str), "'id' is not a string"
        assert isinstance(item["lab_name"], str), "'lab_name' is not a string"
