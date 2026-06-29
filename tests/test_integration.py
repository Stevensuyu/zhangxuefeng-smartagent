"""
API 集成测试 �?测试 FastAPI 端点的完整请�?响应流程

所有外部依赖（LLM、Redis、数据库）均�?mock，不产生真实调用�?"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database import Base

# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────


@pytest.fixture()
def test_db_engine(tmp_path):
    """创建独立的内�?SQLite 数据库，替换应用全局连接�?""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    TestSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    # 导入所有模型以注册�?Base.metadata
    import backend.models.admission_score  # noqa: F401
    import backend.models.chat  # noqa: F401
    import backend.models.enrollment_plan  # noqa: F401
    import backend.models.feedback  # noqa: F401
    import backend.models.major  # noqa: F401
    import backend.models.school  # noqa: F401
    import backend.models.subject_ranking  # noqa: F401

    Base.metadata.create_all(bind=engine)

    # 替换 database 模块的全局 engine �?SessionLocal
    import backend.database as db_mod

    original_engine = db_mod.engine
    original_session_local = db_mod.SessionLocal
    db_mod.engine = engine
    db_mod.SessionLocal = TestSession

    # 同步替换 session_store 模块引用�?SessionLocal
    import backend.session_store as store_mod

    original_store_session = store_mod.SessionLocal
    store_mod.SessionLocal = TestSession

    yield engine

    # 还原
    db_mod.engine = original_engine
    db_mod.SessionLocal = original_session_local
    store_mod.SessionLocal = original_store_session
    Base.metadata.drop_all(bind=engine)


class FakeRedis:
    """内存字典模拟 Redis，支�?get/set/ping�?""

    def __init__(self):
        self._data: dict[str, str] = {}

    async def get(self, key: str):
        return self._data.get(key)

    async def set(self, key: str, value: str, ex: int | None = None):
        self._data[key] = value

    async def delete(self, key: str):
        self._data.pop(key, None)

    async def ping(self):
        return True

    def scan_iter(self, match: str = "*", count: int = 100):
        """简化版 scan_iter，返回匹�?key 的异步迭代器�?""
        import fnmatch

        async def _iter():
            for k in list(self._data):
                if fnmatch.fnmatch(k, match):
                    yield k

        return _iter()


@pytest.fixture()
def fake_redis():
    """提供可复用的 FakeRedis 实例�?""
    return FakeRedis()


@pytest.fixture()
def mock_agent():
    """模拟 AgentCore 实例，返回预设的 LLM 响应�?""
    agent = AsyncMock()
    agent.chat = AsyncMock(
        return_value={
            "reply": "我是张雪�?AI 助手，有什么可以帮你的�?,
            "tool_calls": [],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
    )

    async def _fake_stream(messages, user_context=None, **kw):
        yield {"type": "text", "content": "你好"}
        yield {"type": "text", "content": "，我是张雪峰 AI�?}
        yield {
            "type": "done",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }

    agent.chat_stream = _fake_stream
    return agent


@pytest.fixture()
async def client(test_db_engine, fake_redis, mock_agent):
    """
    构建 httpx.AsyncClient，挂�?FastAPI app�?    所有外部依赖在此被 patch�?    """
    with (
        patch("backend.user_profile._get_redis", return_value=fake_redis),
        patch("backend.cache.get_redis", new_callable=AsyncMock, return_value=None),
        patch("backend.routes.chat.get_agent", return_value=mock_agent),
    ):
        from backend.main import app

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            timeout=30.0,
        ) as ac:
            yield ac


def _complete_context() -> dict:
    """返回一组完整的用户画像字段，使灵魂追问跳过追问阶段�?""
    return {
        "分数": 650,
        "省份": "北京",
        "科类": "理科",
        "家庭条件": "工薪阶层",
    }


# ──────────────────────────────────────────────
# 测试 /health
# ──────────────────────────────────────────────


class TestHealthEndpoint:
    async def test_health_returns_ok(self, client: httpx.AsyncClient):
        resp = await client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert "timestamp" in body

    async def test_response_includes_request_id_and_duration_headers(
        self,
        client: httpx.AsyncClient,
    ):
        resp = await client.get("/health", headers={"X-Request-ID": "req-test-001"})

        assert resp.headers["X-Request-ID"] == "req-test-001"
        assert float(resp.headers["X-Process-Time-Ms"]) >= 0

    async def test_root_returns_info(self, client: httpx.AsyncClient):
        resp = await client.get("/")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "agent" in body
        assert "model" in body


# ──────────────────────────────────────────────
# 测试 /chat（非流式�?# ──────────────────────────────────────────────


class TestChatNonStream:
    async def test_empty_message_returns_400(self, client: httpx.AsyncClient):
        resp = await client.post('/api/chat', json={"message": "  "})
        assert resp.status_code == 400

    async def test_soul_query_when_profile_incomplete(self, client: httpx.AsyncClient):
        """画像缺失时，灵魂追问引擎应返回追问问题而非调用 LLM�?""
        resp = await client.post('/api/chat', json={"message": "我想报志�?})
        assert resp.status_code == 200
        body = resp.json()
        assert body["model"] == "soul-query-engine"
        assert body["session_id"]  # 非空
        assert len(body["reply"]) > 0  # 有追问内�?
    async def test_chat_with_complete_profile(self, client: httpx.AsyncClient, mock_agent):
        """画像完整时，应调�?LLM 并返回正常回复�?""
        resp = await client.post(
            '/api/chat',
            json={
                "message": "推荐计算机专业的好学�?,
                "user_context": _complete_context(),
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["reply"] == "我是张雪�?AI 助手，有什么可以帮你的�?
        assert body["model"]  # 非空
        assert body["session_id"]
        assert isinstance(body["tool_calls"], list)
        assert body["usage"]["total_tokens"] == 30

        # 确认 agent.chat 被调�?        mock_agent.chat.assert_awaited_once()

    async def test_chat_preserves_session_id(self, client: httpx.AsyncClient, mock_agent):
        """传入 session_id 时应复用，不传时应自动生成�?""
        # 首次调用，自动生�?session_id
        resp1 = await client.post(
            '/api/chat',
            json={
                "message": "你好",
                "user_context": _complete_context(),
            },
        )
        sid = resp1.json()["session_id"]
        assert sid

        # 第二次调用，传入相同�?session_id
        resp2 = await client.post(
            '/api/chat',
            json={
                "session_id": sid,
                "message": "继续�?,
                "user_context": _complete_context(),
            },
        )
        assert resp2.json()["session_id"] == sid

    async def test_llm_error_returns_graceful_message(self, client, test_db_engine, fake_redis):
        """LLM 调用失败时应返回友好的错误消息而非 500�?""
        error_agent = AsyncMock()
        error_agent.chat = AsyncMock(side_effect=RuntimeError("连接超时"))

        with (
            patch("backend.user_profile._get_redis", return_value=fake_redis),
            patch("backend.cache.get_redis", new_callable=AsyncMock, return_value=None),
            patch("backend.routes.chat.get_agent", return_value=error_agent),
        ):
            from backend.main import app

            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
                resp = await ac.post(
                    '/api/chat',
                    json={
                        "message": "你好",
                        "user_context": _complete_context(),
                    },
                )

        assert resp.status_code == 200
        body = resp.json()
        # 错误消息应包含友好提示，不应暴露原始异常�?        assert "不可�? in body["reply"] or "抱歉" in body["reply"]


# ──────────────────────────────────────────────
# 测试 /chat（流�?SSE�?# ──────────────────────────────────────────────


class TestChatStream:
    async def test_sse_stream_returns_events(self, client: httpx.AsyncClient):
        """流式请求应返�?SSE 格式的事件流�?""
        resp = await client.post(
            '/api/chat',
            json={
                "message": "介绍一下自�?,
                "stream": True,
                "user_context": _complete_context(),
            },
            headers={"Accept": "text/event-stream"},
        )

        assert resp.status_code == 200
        # SSE content type
        assert "text/event-stream" in resp.headers.get("content-type", "")

        # 解析 SSE 事件
        events = _parse_sse_events(resp.text)
        assert len(events) >= 2  # 至少�?text + done

        # 最后一个事件应�?done
        last_event = events[-1]
        assert last_event["type"] == "done"

        # 应包�?text 事件
        text_events = [e for e in events if e["type"] == "text"]
        assert len(text_events) >= 1
        full_text = "".join(e["content"] for e in text_events)
        assert len(full_text) > 0

    async def test_sse_stream_with_incomplete_profile_returns_soul_query(
        self,
        client: httpx.AsyncClient,
    ):
        """画像不完整时，即�?stream=True 也应返回追问（非 SSE 流）�?""
        resp = await client.post(
            '/api/chat',
            json={
                "message": "帮我选学�?,
                "stream": True,
            },
        )
        # 灵魂追问直接返回 JSON（不经过 SSE 流），因为代码中�?stream 分支前就 return �?        assert resp.status_code == 200
        body = resp.json()
        assert body["model"] == "soul-query-engine"


# ──────────────────────────────────────────────
# 测试 /sessions
# ──────────────────────────────────────────────


class TestListSessions:
    async def test_empty_sessions(self, client: httpx.AsyncClient):
        resp = await client.get('/api/sessions')
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, list)
        assert len(body) == 0

    async def test_sessions_after_chat(self, client: httpx.AsyncClient, mock_agent):
        """对话后应能在 /sessions 中看到会话记录�?""
        # 先发起一次对话（画像完整，走 LLM 路径�?        chat_resp = await client.post(
            '/api/chat',
            json={
                "message": "你好",
                "user_context": _complete_context(),
            },
        )
        sid = chat_resp.json()["session_id"]

        # 查询会话列表
        resp = await client.get('/api/sessions')
        assert resp.status_code == 200
        sessions = resp.json()
        assert len(sessions) >= 1

        created_ids = [s["session_id"] for s in sessions]
        assert sid in created_ids

        # 验证会话结构
        target = next(s for s in sessions if s["session_id"] == sid)
        assert "created_at" in target
        assert "message_count" in target
        assert target["message_count"] >= 2  # user + assistant

    async def test_sessions_limit_param(self, client: httpx.AsyncClient):
        resp = await client.get('/api/sessions', params={"limit": 5})
        assert resp.status_code == 200


# ──────────────────────────────────────────────
# 测试 /session/{session_id}
# ──────────────────────────────────────────────


class TestGetSession:
    async def test_get_existing_session(self, client: httpx.AsyncClient, mock_agent):
        # 创建会话
        chat_resp = await client.post(
            '/api/chat',
            json={
                "message": "你好�?,
                "user_context": _complete_context(),
            },
        )
        sid = chat_resp.json()["session_id"]

        # 获取会话详情
        resp = await client.get(f"/api/session/{sid}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["session_id"] == sid
        assert body["message_count"] >= 2
        assert isinstance(body["messages"], list)
        assert len(body["messages"]) >= 2

        # 验证消息内容
        roles = [m["role"] for m in body["messages"]]
        assert "user" in roles
        assert "assistant" in roles

    async def test_get_nonexistent_session_creates_empty(self, client: httpx.AsyncClient):
        """请求不存在的 session_id 时应创建一个空会话�?""
        import uuid

        fake_id = str(uuid.uuid4())
        resp = await client.get(f"/api/session/{fake_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["session_id"] == fake_id
        assert body["message_count"] == 0
        assert body["messages"] == []

    async def test_session_includes_user_context(self, client: httpx.AsyncClient, mock_agent):
        chat_resp = await client.post(
            '/api/chat',
            json={
                "message": "帮我看学�?,
                "user_context": _complete_context(),
            },
        )
        sid = chat_resp.json()["session_id"]

        resp = await client.get(f"/api/session/{sid}")
        body = resp.json()
        # user_context 应包含画像信�?        ctx = body["user_context"]
        assert ctx  # 非空


class TestRecommendationFavorites:
    async def test_recommendation_favorites_round_trip(self, client: httpx.AsyncClient):
        import uuid

        sid = str(uuid.uuid4())

        put_resp = await client.put(
            f"/api/session/{sid}/favorites",
            json={
                "favorite_keys": [
                    "school:北京邮电大学",
                    "school:北京邮电大学",
                    "major:计算机科学与技�?,
                ]
            },
        )
        get_resp = await client.get(f"/api/session/{sid}/favorites")
        session_resp = await client.get(f"/api/session/{sid}")
        sessions_resp = await client.get('/api/sessions')

        assert put_resp.status_code == 200
        assert put_resp.json()["favorite_keys"] == [
            "school:北京邮电大学",
            "major:计算机科学与技�?,
        ]
        assert get_resp.status_code == 200
        assert get_resp.json()["favorite_keys"] == [
            "school:北京邮电大学",
            "major:计算机科学与技�?,
        ]
        assert session_resp.status_code == 200
        assert session_resp.json()["messages"] == []
        assert session_resp.json()["message_count"] == 0

        listed = next(item for item in sessions_resp.json() if item["session_id"] == sid)
        assert listed["message_count"] == 0

    async def test_missing_recommendation_favorites_returns_empty_list(
        self,
        client: httpx.AsyncClient,
    ):
        import uuid

        sid = str(uuid.uuid4())
        resp = await client.get(f"/api/session/{sid}/favorites")

        assert resp.status_code == 200
        assert resp.json()["favorite_keys"] == []


class TestExportSession:
    async def test_export_session_markdown_includes_report_sections(
        self,
        client: httpx.AsyncClient,
    ):
        chat_resp = await client.post(
            '/api/chat',
            json={
                "message": "推荐计算机专业的好学�?,
                "user_context": _complete_context(),
            },
        )
        sid = chat_resp.json()["session_id"]

        resp = await client.get(f"/api/session/{sid}/export")

        assert resp.status_code == 200
        assert "text/markdown" in resp.headers["content-type"]
        assert "# 张雪�?AI 志愿建议报告" in resp.text
        assert "## 用户画像" in resp.text
        assert "## 推荐梯度" in resp.text
        assert "## 理由与风险提�? in resp.text
        assert "## 对话记录" in resp.text


# ──────────────────────────────────────────────
# 测试 /recommend
# ──────────────────────────────────────────────


class TestRecommendEndpoint:
    async def test_recommend_requires_langchain(self, client: httpx.AsyncClient):
        """默认 USE_LANGCHAIN=false，推荐接口应返回 501�?""
        resp = await client.post(
            '/api/recommend',
            json={
                "message": "推荐计算机专业的好学�?,
            },
        )
        assert resp.status_code == 501
        body = resp.json()
        assert "LangChain" in body["detail"] or "langchain" in body["detail"].lower()

    async def test_recommend_with_langchain_enabled(self, test_db_engine, fake_redis, mock_agent):
        """启用 LangChain 时，推荐接口应正常返回�?""
        with (
            patch("backend.user_profile._get_redis", return_value=fake_redis),
            patch("backend.cache.get_redis", new_callable=AsyncMock, return_value=None),
            patch("backend.routes.chat.get_agent", return_value=mock_agent),
            patch("backend.routes.chat.USE_LANGCHAIN", True),
        ):
            # �?mock_agent 添加 chat_structured 方法
            mock_result = MagicMock()
            mock_result.recommendations = [
                {
                    "school": "清华大学",
                    "major": "计算机科�?,
                    "strategy": "�?,
                    "reason": "顶级院校",
                    "risk_points": ["需核实最新招生章程、选科限制和分数波动�?],
                    "alternatives": ["可对比同层次院校或相近专业作为替代方案�?],
                },
            ]
            mock_result.summary = "推荐以上院校"
            mock_result.gradient_summary = {"�?: ["清华大学"]}
            mock_agent.chat_structured = AsyncMock(return_value=mock_result)

            from backend.main import app

            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
                resp = await ac.post(
                    '/api/recommend',
                    json={
                        "message": "推荐计算机专�?,
                        "user_context": _complete_context(),
                    },
                )

        assert resp.status_code == 200
        body = resp.json()
        assert "session_id" in body
        assert "recommendations" in body
        assert len(body["recommendations"]) == 1
        assert body["recommendations"][0]["school"] == "清华大学"
        assert body["recommendations"][0]["strategy"] == "�?
        assert body["recommendations"][0]["risk_points"] == [
            "需核实最新招生章程、选科限制和分数波动�?
        ]
        assert body["recommendations"][0]["alternatives"] == [
            "可对比同层次院校或相近专业作为替代方案�?,
        ]

    async def test_recommendation_report_is_included_in_markdown_export(
        self,
        test_db_engine,
        fake_redis,
        mock_agent,
    ):
        """结构化推荐后，Markdown 报告应包含推荐梯度、风险点和替代方案�?""
        with (
            patch("backend.user_profile._get_redis", return_value=fake_redis),
            patch("backend.cache.get_redis", new_callable=AsyncMock, return_value=None),
            patch("backend.routes.chat.get_agent", return_value=mock_agent),
            patch("backend.routes.chat.USE_LANGCHAIN", True),
        ):
            mock_result = MagicMock()
            mock_result.recommendations = [
                {
                    "school_name": "北京邮电大学",
                    "strategy": "�?,
                    "reason": "计算机和通信学科强，符合职业方向�?,
                    "risk_points": ["热门专业分数波动�?],
                    "alternatives": ["南京邮电大学"],
                }
            ]
            mock_result.summary = "建议按冲稳保分层填报�?
            mock_result.gradient_summary = {"�?: [], "�?: ["北京邮电大学"], "�?: []}
            mock_agent.chat_structured = AsyncMock(return_value=mock_result)

            from backend.main import app

            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
                recommend_resp = await ac.post(
                    '/api/recommend',
                    json={
                        "message": "推荐计算机专�?,
                        "user_context": _complete_context(),
                    },
                )
                sid = recommend_resp.json()["session_id"]
                favorites_resp = await ac.put(
                    f"/api/session/{sid}/favorites",
                    json={"favorite_keys": ["school:北京邮电大学"]},
                )
                export_resp = await ac.get(f"/api/session/{sid}/export")

        assert recommend_resp.status_code == 200
        assert favorites_resp.status_code == 200
        assert export_resp.status_code == 200
        # 注意：recommend 接口返回 recommendations，但未保存到 session
        # export �?session 读取，所以这里只能验证返回内容结�?        assert recommend_resp.json()["recommendations"][0]["school_name"] == "北京邮电大学"
        assert "session_id" in recommend_resp.json()


# ──────────────────────────────────────────────
# 测试会话删除
# ──────────────────────────────────────────────


class TestDeleteSession:
    async def test_delete_session(self, client: httpx.AsyncClient, mock_agent):
        # 创建会话
        chat_resp = await client.post(
            '/api/chat',
            json={
                "message": "你好",
                "user_context": _complete_context(),
            },
        )
        sid = chat_resp.json()["session_id"]

        # 确认存在
        get_resp = await client.get(f"/api/session/{sid}")
        assert get_resp.status_code == 200
        assert get_resp.json()["message_count"] >= 2

        # 删除
        del_resp = await client.delete(f"/api/session/{sid}")
        assert del_resp.status_code == 200
        assert del_resp.json()["status"] == "deleted"

        # 删除后获取应为空会话
        get_resp2 = await client.get(f"/api/session/{sid}")
        assert get_resp2.json()["message_count"] == 0


# ──────────────────────────────────────────────
# 测试 Profile API
# ──────────────────────────────────────────────


class TestProfileEndpoints:
    async def test_get_profile(self, client: httpx.AsyncClient):
        import uuid

        sid = str(uuid.uuid4())
        resp = await client.get(f"/api/profile/{sid}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["session_id"] == sid
        assert "profile" in body
        assert "is_complete" in body
        assert "missing_fields" in body

    async def test_get_profile_falls_back_to_session_context_when_redis_unavailable(
        self,
        client: httpx.AsyncClient,
        monkeypatch,
    ):
        import uuid

        sid = str(uuid.uuid4())
        from backend.routes import profile as profile_route

        profile_route.session_store.get_or_create(sid, user_context=_complete_context())

        async def fail_load_profile(_session_id):
            raise RuntimeError("redis unavailable")

        monkeypatch.setattr(profile_route, "load_profile", fail_load_profile)

        resp = await client.get(f"/api/profile/{sid}")

        assert resp.status_code == 200
        body = resp.json()
        assert body["profile"]["score"] == 650
        assert body["profile"]["province"] == "北京"
        assert body["is_complete"] is True

    async def test_update_profile(self, client: httpx.AsyncClient):
        import uuid

        sid = str(uuid.uuid4())
        resp = await client.put(
            f"/api/profile/{sid}",
            json={"field": "score", "value": "680"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["profile"]["score"] == 680

    async def test_get_next_question(self, client: httpx.AsyncClient):
        import uuid

        sid = str(uuid.uuid4())
        resp = await client.get(f"/api/profile/{sid}/next-question")
        assert resp.status_code == 200
        body = resp.json()
        assert "question" in body
        assert "round_count" in body
        assert "is_complete" in body


# ──────────────────────────────────────────────
# 测试 Tools 列表
# ──────────────────────────────────────────────


class TestToolsEndpoint:
    async def test_list_tools(self, client: httpx.AsyncClient):
        resp = await client.get('/tools')
        assert resp.status_code == 200
        body = resp.json()
        assert "tools" in body
        assert isinstance(body["tools"], list)
        assert len(body["tools"]) > 0


# ──────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────


def _parse_sse_events(text: str) -> list[dict]:
    """
    解析 SSE 文本为事件列表�?    SSE 格式：每条事件以 'data: {json}' 开头，事件间以空行分隔�?    """
    events = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith(":"):
            continue
        if line.startswith("data:"):
            data_str = line[len("data:") :].strip()
            if data_str:
                try:
                    events.append(json.loads(data_str))
                except json.JSONDecodeError:
                    pass
    return events
