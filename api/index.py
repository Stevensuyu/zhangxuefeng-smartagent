"""
Vercel Serverless Function 入口 - 张雪峰 AI Agent
"""

import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from dotenv import load_dotenv

load_dotenv(override=True)

from mangum import Mangum
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 创建 FastAPI 应用
app = FastAPI(
    title="张雪峰 AI 咨询 Agent",
    description="高考/考研/职业规划咨询，基于张雪峰认知操作系统",
    version="0.2.0",
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 导入后端路由
try:
    from backend.main import app as main_app
    
    # 复制所有路由
    for route in main_app.routes:
        app.routes.append(route)
except ImportError as e:
    print(f"Warning: Could not import backend routes: {e}")

# SSE 流式输出路由
try:
    from backend.routes.chat import router as chat_router
    app.include_router(chat_router, prefix="/api")
except ImportError:
    pass

# 会话路由
try:
    from backend.routes.session import router as session_router
    app.include_router(session_router, prefix="/api")
except ImportError:
    pass

# 画像路由
try:
    from backend.routes.profile import router as profile_router
    app.include_router(profile_router, prefix="/api")
except ImportError:
    pass

# 系统路由
try:
    from backend.routes.system import router as system_router
    app.include_router(system_router)
except ImportError:
    pass

# 数据查询路由
try:
    from backend.routers import schools_router, majors_router, scores_router, plans_router, subject_rankings_router
    app.include_router(schools_router, prefix="/api")
    app.include_router(majors_router, prefix="/api")
    app.include_router(scores_router, prefix="/api")
    app.include_router(plans_router, prefix="/api")
    app.include_router(subject_rankings_router, prefix="/api")
except ImportError:
    pass

# 健康检查端点
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "zhangxuefeng-agent"}

@app.get("/")
async def root():
    return {
        "name": "张雪峰 AI 咨询 Agent",
        "version": "0.2.0",
        "description": "高考/考研/职业规划咨询"
    }

# Vercel Serverless Handler
handler = Mangum(app, lifespan="off")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
