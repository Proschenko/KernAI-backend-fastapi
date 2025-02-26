from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from .schemas import Laboratories
from datetime import date



async def get_labs(session: AsyncSession)-> list[Laboratories]:
    query=text("""
        SELECT id, labname
        FROM public.laboratories""")
    
    result = await session.execute(query)
    labs_data =  result.fetchall()
    return [Laboratories(**row._mapping) for row in labs_data]