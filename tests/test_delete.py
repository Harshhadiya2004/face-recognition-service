from unittest.mock import patch
from tests.mock_milvus import MockMilvus
import pytest

@patch("app.main.get_milvus_client")
@pytest.mark.asyncio
async def test_delete(mock_milvus, async_client):

    mock_milvus.return_value = MockMilvus()

    headers = {
        "X-Username": "ORG1",
        "X-Authenticated-By": "nginx-gateway"
    }

    res = await async_client.delete(
        "/api/face/USER105",
        headers=headers
    )

    assert res.status_code == 200