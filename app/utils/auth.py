from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from keycloak import KeycloakOpenID
import os

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
    client_secret_key=KEYCLOAK_CLIENT_SECRET
)

# Используем  Bearer
security = HTTPBearer()

async def decode_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    session: AsyncSession = Depends(get_session),
):
    """Проверка токена через Keycloak и сохранение пользователя в БД"""
    token = credentials.credentials
    try:
        user_info = keycloak_openid.userinfo(token)
        if not user_info:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        #username = user_info.get("preferred_username", "Unknown User")
        email = user_info.get("email", "No Email") # !Имя пользователя и есть его почта!
        # Получаем ID пользователя (если нет, то создаем)
        user_id = await serv.check_and_add_user(session, username=email)

        return {"username": email, "id": user_id}


    except Exception as e:
        print(f"Ошибка проверки токена: {str(e)}")
        raise HTTPException(status_code=401, detail="Token validation failed")

def check_role(required_role: str):
    """Функция-декоратор для проверки ролей"""
    def role_checker(user=Depends(decode_token)):
        roles = user["user_info"].get("realm_access", {}).get("roles", [])
        if required_role not in roles:
            raise HTTPException(status_code=403, detail=f"Access denied. Required role: {required_role}")
        return user
    return role_checker
