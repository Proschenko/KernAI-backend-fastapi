import pytest
import httpx
import random

API_KERNS_URL = "http://127.0.0.1:8000/kerns"
API_KERN_COMMENTS_URL = "http://127.0.0.1:8000/kern/{id}/comments"

@pytest.mark.asyncio
async def test_get_kern_comments_structure():
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
        
        # Используем полученный ID для запроса на /kern/{id}/comments
        comments_response = await client.get(API_KERN_COMMENTS_URL.format(id=kern_id))
        assert comments_response.status_code == 200, f"Expected status 200, got {comments_response.status_code}"
        
        # Проверяем структуру ответа для комментариев
        comments_json = comments_response.json()
        assert isinstance(comments_json, list), "Response is not a list"
        
        # Проверяем, что каждый комментарий имеет правильную структуру
        for comment in comments_json:
            assert isinstance(comment, dict), "Item is not a dictionary"
            assert "id" in comment, "Missing 'id' in comment"
            assert "insert_date" in comment, "Missing 'insert_date' in comment"
            assert "insert_user" in comment, "Missing 'insert_user' in comment"
            assert "comment_text" in comment, "Missing 'comment_text' in comment"
            assert "kern_code" in comment, "Missing 'kern_code' in comment"
            assert "lab_name" in comment, "Missing 'lab_name' in comment"
            
            assert isinstance(comment["id"], str), "'id' is not a string"
            assert isinstance(comment["insert_date"], str), "'insert_date' is not a string"
            assert isinstance(comment["insert_user"], str), "'insert_user' is not a string"
            assert isinstance(comment["comment_text"], str), "'comment_text' is not a string"
            assert isinstance(comment["kern_code"], str), "'kern_code' is not a string"
            assert isinstance(comment["lab_name"], str), "'lab_name' is not a string"
