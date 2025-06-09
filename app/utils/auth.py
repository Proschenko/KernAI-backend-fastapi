from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from keycloak import KeycloakOpenID
import os
import json
import base64

from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_session  # Импортируем создание сессии
import app.service as serv 

# Загружаем конфиг из .env
KEYCLOAK_SERVER_URL = os.getenv("KEYCLOAK_SERVER_URL")
KEYCLOAK_REALM = os.getenv("KEYCLOAK_REALM")
KEYCLOAK_CLIENT_ID = os.getenv("KEYCLOAK_CLIENT_ID")
KEYCLOAK_CLIENT_SECRET = os.getenv("KEYCLOAK_CLIENT_SECRET")

# Создаем клиент Keycloak
keycloak_openid = KeycloakOpenID(
    server_url=KEYCLOAK_SERVER_URL,
    client_id=KEYCLOAK_CLIENT_ID,
    realm_name=KEYCLOAK_REALM,
    client_secret_key=KEYCLOAK_CLIENT_SECRET,
    # verify="/ssl/kc_root_crt.pem" 
    verify = False #отключить проверку https
)

# Используем Bearer
security = HTTPBearer()

def decode_jwt(token: str) -> dict:
    """Декодирование JWT-токена без верификации"""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("Неверный формат JWT токена")

        payload_b64 = parts[1]
        payload_b64 += "=" * (4 - len(payload_b64) % 4)  # Исправляем padding
        payload_decoded = base64.urlsafe_b64decode(payload_b64).decode("utf-8")
        
        return json.loads(payload_decoded)
    except Exception as e:
        print(f"Ошибка декодирования JWT: {str(e)}")
        return {}

async def decode_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    session: AsyncSession = Depends(get_session),
):
    """Проверка токена через Keycloak и сохранение пользователя в БД"""
    token = credentials.credentials
    try:
        # Получаем информацию о пользователе из Keycloak
        user_info = keycloak_openid.userinfo(token)
        if not user_info:
            raise HTTPException(status_code=401, detail="Invalid token")

        email = user_info.get("email", "No Email")  # В системе email = username
        user_id = await serv.check_and_add_user(session, username=email)

        # Декодируем токен вручную через Base64
        token_decoded = decode_jwt(token)

        # Извлекаем роли из kern-ai-front
        roles = token_decoded.get("resource_access", {}).get("kern-ai-front", {}).get("roles", [])

        return {
            "username": email,
            "id": user_id,
            "roles": roles,
            "token_decoded": token_decoded  # Полный расшифрованный токен
        }

    except Exception as e:
        print(f"Ошибка проверки токена: {str(e)}")
        raise HTTPException(status_code=401, detail="Token validation failed")

def check_role(required_role: str):
    """Функция-декоратор для проверки ролей"""
    def role_checker(user=Depends(decode_token)):
        if required_role not in user["roles"]:
            raise HTTPException(status_code=403, detail=f"Access denied. Required role: {required_role}")
        return user
    return role_checker