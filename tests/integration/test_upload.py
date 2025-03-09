# tests/integration/test_upload.py
import pytest

@pytest.mark.asyncio
async def test_upload_image(client):
    with open(r"C:\Users\Prosc\PycharmProjects\KernAI\KernAI-backend-v1\tests\.test_static\0001.jpg", "rb") as img_file:
        response = client.post("/upload_img", files={"file": img_file}, params={"username": "test_user"})
        assert response.status_code == 200
        assert "filename" in response.json()
        assert "file_path" in response.json()
