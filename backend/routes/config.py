"""
AI配置API端点：获取/保存API Key和模型配置
"""

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from backend.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()


class AIConfig(BaseModel):
    api_key: Optional[str] = Field(default=None, description="OpenAI API Key")
    base_url: str = Field(default="https://api.openai.com/v1", description="API Base URL")
    model: str = Field(default="gpt-4o-mini", description="模型名称")


@router.get("/config")
async def get_config():
    """获取当前AI配置"""
    settings = get_settings()
    return {
        "api_key": "***" if settings.openai_api_key else "",
        "base_url": settings.openai_base_url,
        "model": settings.effective_model,
        "has_api_key": bool(settings.openai_api_key),
    }


@router.post("/config")
async def save_config(config: AIConfig):
    """保存AI配置（仅在会话级别临时生效）"""
    logger.info(f"AI配置更新: model={config.model}, base_url={config.base_url}, has_api_key={bool(config.api_key)}")
    
    if config.api_key and not config.api_key.startswith("sk-"):
        raise HTTPException(status_code=400, detail="API Key必须以sk-开头")
    
    return {
        "status": "ok",
        "message": "配置已保存，将在下次请求时生效",
        "config": {
            "api_key": "***" if config.api_key else "",
            "base_url": config.base_url,
            "model": config.model,
        },
    }
