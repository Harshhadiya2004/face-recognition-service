import io
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.main import app

client = TestClient(app)

@pytest.fixture
def mock_face_detection():
    return [
    {
    "embedding": [0.1] * 128,
    "bbox": [10, 20, 110, 120],
    "confidence": 0.95
    }
    ]

@pytest.fixture
def mock_db():
    db = MagicMock()
    db.add = MagicMock()
    db.commit = MagicMock()
    return db

@patch("app.main.SessionLocal")
@patch("app.main.get_milvus_client")
@patch("app.main.extract_multiple_faces")
def test_identify_multiple_faces_success(
mock_extract,
mock_milvus,
mock_session,
mock_face_detection,
mock_db
):
    # Mock database session
    mock_session.return_value = mock_db

    # Mock face detection
    mock_extract.return_value = mock_face_detection

    # Mock Milvus search result
    mock_milvus_instance = MagicMock()
    mock_milvus_instance.search_similar.return_value = [("USER101", 0.95)]
    mock_milvus.return_value = mock_milvus_instance

    file = {
        "file": ("test.jpg", io.BytesIO(b"fake_image_data"), "image/jpeg")
    }

    response = client.post(
        "/api/face/identify-multiple",
        files=file,
        data={
            "camera_id": "Cam_1",
            "flag": "lecture"
        },
        headers={
            "X-Username": "Org_1",
            "X-Authenticated-By": "nginx-gateway"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["total_faces_detected"] == 1
    assert data["identified_faces"][0]["match"] is True
    assert data["identified_faces"][0]["user_id"] == "USER101"
    assert data["camera_id"] == "Cam_1"


@patch("app.main.extract_multiple_faces")
def test_identify_multiple_faces_no_faces(mock_extract):
    mock_extract.return_value = []

    file = {
        "file": ("test.jpg", io.BytesIO(b"fake_image_data"), "image/jpeg")
    }

    response = client.post(
        "/api/face/identify-multiple",
        files=file,
        headers={
            "X-Username": "Org_1",
            "X-Authenticated-By": "nginx-gateway"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["total_faces_detected"] == 0
    assert data["identified_faces"] == []


def test_identify_multiple_faces_unauthorized():

    file = {
        "file": ("test.jpg", io.BytesIO(b"fake_image_data"), "image/jpeg")
    }

    response = client.post(
        "/api/face/identify-multiple",
        files=file
    )

    assert response.status_code == 400

@patch("app.main.get_milvus_client")
@patch("app.main.extract_multiple_faces")
def test_identify_multiple_faces_no_match(
mock_extract,
mock_milvus,
mock_face_detection
):

    mock_extract.return_value = mock_face_detection

    mock_milvus_instance = MagicMock()
    mock_milvus_instance.search_similar.return_value = []
    mock_milvus.return_value = mock_milvus_instance

    file = {
        "file": ("test.jpg", io.BytesIO(b"fake_image_data"), "image/jpeg")
    }

    response = client.post(
        "/api/face/identify-multiple",
        files=file,
        headers={
            "X-Username": "Org_1",
            "X-Authenticated-By": "nginx-gateway"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["identified_faces"][0]["match"] is False
