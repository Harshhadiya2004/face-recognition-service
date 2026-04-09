from unittest.mock import patch
from tests.mock_milvus import MockMilvus
import pytest

@patch("app.main.get_milvus_client")
@pytest.mark.asyncio
async def test_health(mock_milvus, async_client):

    mock_milvus.return_value = MockMilvus()

    res = await async_client.get("/health")

    assert res.status_code == 200
    assert res.json()["status"] == "healthy"