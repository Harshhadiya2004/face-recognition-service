# Prayas Face Recognition Service

A FastAPI-based face recognition service for user attendance tracking. Uses InsightFace for face feature extraction and Milvus vector database for efficient similarity search.

## Features

- **Face Enrollment**: Register user faces with their IDs
- **Face Identification**: Identify users from uploaded images
- **Vector Database**: Milvus-backed storage for face embeddings
- **Gateway Authentication**: Protected via Nginx API Gateway with JWT
- **Organization Isolation**: Multi-tenant support with organization-based filtering
- **Docker Support**: Containerized deployment ready

## Tech Stack

- **FastAPI** - Modern Python web framework
- **InsightFace** - State-of-the-art face recognition library
- **ONNX Runtime** - Efficient model inference
- **Milvus** - Vector database for similarity search
- **OpenCV** - Image processing
- **Uvicorn** - ASGI server

## Prerequisites

- Docker and Docker Compose
- Milvus vector database (running separately or via docker-compose)
- Python 3.10+ (for local development)

## Quick Start

### Using Docker

```bash
# Build the image
docker build -t face-recognition-service .

# Run the container
docker run -p 8001:8001 \
  -e MILVUS_HOST=milvus \
  -e MILVUS_PORT=19530 \
  face-recognition-service
```

### With Docker Compose

See the parent `docker-compose.yml` for full stack deployment including:
- Nginx API Gateway
- Milvus vector database
- This face recognition service

```bash
docker-compose up -d face-recognition-service
```

## API Endpoints

### Public Endpoints

#### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "Prayas Face Recognition",
  "milvus_connected": true,
  "collection_stats": {
    "num_entities": 150,
    "name": "face_embeddings"
  }
}
```

### Protected Endpoints (requires JWT via Gateway)

#### `POST /api/face/enroll`
Enroll a new face for a user.

**Headers:**
- `X-Authenticated-By: nginx-gateway` (injected by gateway)
- `X-Username`: Organization name (injected by gateway from JWT)

**Form Data:**
- `file`: Image file containing the face
- `user_id`: User identifier

**Response:**
```json
{
  "success": true,
  "user_id": "USER001",
  "message": "Face enrolled successfully"
}
```

**Error Responses:**
- `409 USER_ALREADY_EXISTS`: User already has a face registered in this organization
- `409 FACE_ALREADY_ENROLLED`: Face is already registered for another user

#### `PUT /api/face/update`
Update face embedding for an existing user.

**Form Data:**
- `file`: Image file containing the new face
- `user_id`: User identifier to update

**Response:**
```json
{
  "success": true,
  "user_id": "USER001",
  "message": "Face updated successfully",
  "previous_embedding_deleted": true
}
```

#### `POST /api/face/identify`
Identify a face from an uploaded image.

**Headers:**
- `X-Authenticated-By: nginx-gateway` (injected by gateway)
- `X-Username`: Organization name (injected by gateway from JWT)

**Form Data:**
- `file`: Image file containing the face

**Response:**
```json
{
  "match": true,
  "user_id": "USER001",
  "similarity": 0.92,
  "threshold": 0.5
}
```

#### `DELETE /api/face/{user_id}`
Delete all face embeddings for a user.

**Response:**
```json
{
  "success": true,
  "user_id": "USER001",
  "deleted_count": 1
}
```

#### `GET /api/face/stats`
Get collection statistics.

**Response:**
```json
{
  "num_entities": 150,
  "name": "face_embeddings"
}
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MILVUS_HOST` | `localhost` | Milvus server host |
| `MILVUS_PORT` | `19530` | Milvus server port |
| `SERVICE_PORT` | `8001` | Service port |
| `THRESHOLD` | `0.6` | Face match similarity threshold |
| `REQUIRE_GATEWAY_AUTH` | `true` | Require gateway authentication |
| `LOG_LEVEL` | `INFO` | Logging level |

## Project Structure

```
face-recognition-service/
├── app/
│   ├── __init__.py
│   └── main.py              # FastAPI application & endpoints
├── utils/
│   ├── __init__.py
│   ├── face_config.py       # Configuration constants
│   ├── face_processor.py    # Face embedding extraction
│   └── milvus_client.py     # Milvus database operations
├── Dockerfile
├── requirements.txt
└── README.md
```

## Development

### Local Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the service
uvicorn app.main:app --reload --port 8001
```

### Testing

```bash
# Health check
curl http://localhost:8001/health

# Enroll a face (with gateway auth)
curl -X POST http://localhost:8001/api/face/enroll \
  -H "X-Authenticated-By: nginx-gateway" \
  -H "X-Username: my-org" \
  -F "file=@face.jpg" \
  -F "user_id=USER001"

# Update a face
curl -X PUT http://localhost:8001/api/face/update \
  -H "X-Authenticated-By: nginx-gateway" \
  -H "X-Username: my-org" \
  -F "file=@new_face.jpg" \
  -F "user_id=USER001"

# Identify a face
curl -X POST http://localhost:8001/api/face/identify \
  -H "X-Authenticated-By: nginx-gateway" \
  -H "X-Username: my-org" \
  -F "file=@test_face.jpg"

# Delete a face
curl -X DELETE http://localhost:8001/api/face/USER001 \
  -H "X-Authenticated-By: nginx-gateway" \
  -H "X-Username: my-org"
```

## License

MIT License - See LICENSE file for details

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
