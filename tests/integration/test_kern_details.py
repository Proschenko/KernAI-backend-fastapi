import pytest
import httpx
import random

API_KERNS_URL = "http://127.0.0.1:8000/kerns"
API_KERN_DETAIL_URL = "http://127.0.0.1:8000/kern/{id}"

@pytest.mark.asyncio
async def test_get_kern_detail_structure():
    async with httpx.AsyncClient() as client:
        # Получаем список всех "kern" объектов
        response = await client.get(API_KERNS_URL)
        assert response.status_code == 200, f"Expected status 200, got {response.status_code}"
        
        # Получаем список и выбираем случайный элемент
        json_response = response.json()
        assert isinstance(json_response, list), "Response is not a list"
        assert len(json_response) > 0, "No kern items found"
        
        # Выбираем случайный элемент и получаем его ID
        random_kern = random.choice(json_response)
        kern_id = random_kern["id"]
        
        # Используем полученный ID для запроса на /kern/{id}
        detail_response = await client.get(API_KERN_DETAIL_URL.format(id=kern_id))
        assert detail_response.status_code == 200, f"Expected status 200, got {detail_response.status_code}"
        
        # Проверяем, что ответ является списком
        detail_json = detail_response.json()
        assert isinstance(detail_json, list), "Response is not a list"
        
        # Проверяем структуру первого элемента в списке
        first_item = detail_json[0]  # Берем первый элемент из списка
        
        assert isinstance(first_item, dict), "First item is not a dictionary"
        assert "id" in first_item, "Missing 'id' in first item"
        assert "insert_user" in first_item, "Missing 'insert_user' in first item"
        assert "insert_date" in first_item, "Missing 'insert_date' in first item"
        assert "lab_name" in first_item, "Missing 'lab_name' in first item"
        assert "kern_code" in first_item, "Missing 'kern_code' in first item"
        assert "damage_type" in first_item, "Missing 'damage_type' in first item"
        
        assert isinstance(first_item["id"], str), "'id' is not a string"
        assert isinstance(first_item["insert_user"], str), "'insert_user' is not a string"
        assert isinstance(first_item["insert_date"], str), "'insert_date' is not a string"
        assert isinstance(first_item["lab_name"], str), "'lab_name' is not a string"
        assert isinstance(first_item["kern_code"], str), "'kern_code' is not a string"
        assert first_item["damage_type"] is None or isinstance(first_item["damage_type"], str), "'damage_type' is not valid"
