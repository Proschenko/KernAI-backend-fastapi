import pytest
import httpx

API_URL = "http://127.0.0.1:8000/kerns"

@pytest.mark.asyncio
async def test_get_kerns_structure():
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
        assert "kern_code" in item, "Missing 'kern_code' in item"
        assert "lab_name" in item, "Missing 'lab_name' in item"
        assert "last_date" in item, "Missing 'last_date' in item"
        assert "user_name" in item, "Missing 'user_name' in item"
        assert "damage_type" in item, "Missing 'damage_type' in item"
        
        assert isinstance(item["id"], str), "'id' is not a string"
        assert isinstance(item["kern_code"], str), "'kern_code' is not a string"
        assert isinstance(item["lab_name"], str), "'lab_name' is not a string"
        assert isinstance(item["last_date"], str), "'last_date' is not a string"
        assert isinstance(item["user_name"], str), "'user_name' is not a string"
        
        # damage_type может быть строкой или None
        assert item["damage_type"] is None or isinstance(item["damage_type"], str), "'damage_type' is not valid"
