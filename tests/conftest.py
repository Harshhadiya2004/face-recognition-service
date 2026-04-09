import pytest_asyncio
from httpx import AsyncClient
from httpx import ASGITransport
from app.main import app
import os

# Disable Nginx Gateway Auth for testing
os.environ["REQUIRE_GATEWAY_AUTH"] = "false"

@pytest_asyncio.fixture
async def async_client():

    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test"
    ) as client:
        yield client