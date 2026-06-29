"""
Vercel Serverless Function 入口 - 适配 FastAPI 应用
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio

from dotenv import load_dotenv

load_dotenv(override=True)

from backend.main import app
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.responses import JSONResponse
from fastapi.requests import Request
from starlette.middleware.wsgi import WSGIMiddleware
from fastapi import FastAPI

try:
    import uvicorn
    from fastapi import FastAPI
    from mangum import Mangum
    
    handler = Mangum(app, lifespan="off")
    
except ImportError:
    async def async_handler(event, context):
        from mangum import Mangum
        handler = Mangum(app, lifespan="off")
        return await handler(event, context)
    
    def handler(event, context):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(async_handler(event, context))
        loop.close()
        return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
