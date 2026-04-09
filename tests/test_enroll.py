from unittest.mock import patch
from tests.mock_milvus import MockMilvus
import pytest
import numpy as np

@patch("app.main.get_milvus_client")
@patch("app.main.extract_embedding")
@pytest.mark.asyncio
async def test_enroll_face(mock_embed, mock_milvus, async_client):

    mock_milvus.return_value = MockMilvus()
    mock_embed.return_value = (np.zeros(512), [0,0,0,0])

    files = {
        "file": ("face.jpg", b"fakeimage", "image/jpeg")
    }

    headers = {
        "X-Username": "ORG1",
        "X-Authenticated-By": "nginx-gateway"
    }

    res = await async_client.post(
        "/api/face/enroll",
        files=files,
        data={"user_id":"USER105"},
        headers=headers
    )

    assert res.status_code == 200
    assert res.json()["success"] == True