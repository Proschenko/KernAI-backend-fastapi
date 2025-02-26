# test_db_connection.py
import logging
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
import os
from dotenv import load_dotenv

# Настройка логирования
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Загрузка переменных окружения
load_dotenv()

# Получение DATABASE_URL
DATABASE_URL = os.getenv("DATABASE_URL")

# Логирование и проверка переменной
if not DATABASE_URL:
    logging.error("DATABASE_URL не найден в переменных окружения!")
    exit(1)

logging.info(f"Попытка подключения к базе данных с URL: {DATABASE_URL}")

# Попытка подключения к базе данных
try:
    # Создание подключения к базе данных
    engine = create_engine(DATABASE_URL)
    
    # Попытка выполнения простого запроса
    with engine.connect() as connection:
        logging.info("Подключение к базе данных успешно!")
except SQLAlchemyError as e:
    logging.error(f"Ошибка подключения к базе данных: {e}")
    exit(1)
