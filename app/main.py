"""
FastAPI face recognition service for Prayas backend.
Provides endpoints for face enrollment and attendance identification.
Uses Milvus vector database for embedding storage and search.
Integrates with Nginx API Gateway for JWT authentication.
"""

import logging
import os
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Header, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional  # Add this line

from utils.face_config import THRESHOLD, SERVICE_PORT
from utils.face_processor import extract_embedding, extract_multiple_faces
from utils.milvus_client import get_milvus_client

from sqlalchemy.orm import Session
from db.database import engine, SessionLocal
from db.models import Base, StudentAttendance, AttendanceType
from datetime import datetime, date

Base.metadata.create_all(bind=engine)

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Prayas Face Recognition Service",
    version="1.0.0",
    description="Face recognition service with Milvus vector database for user attendance",
    docs_url="/face-docs",
    redoc_url="/face-redoc",
    openapi_url="/face-openapi.json"
)

# Add CORS middleHey, Cortana. ware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Response Models
class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    service: str
    milvus_connected: bool
    collection_stats: Optional[dict] = None


class FaceEnrollResponse(BaseModel):
    """Face enrollment response model."""
    success: bool
    user_id: str
    message: str


class FaceIdentifyResponse(BaseModel):
    """Face identification response model."""
    match: bool
    user_id: Optional[str]
    similarity: float
    threshold: float


class CollectionStatsResponse(BaseModel):
    """Collection statistics response model."""
    num_entities: int
    name: str


class DeleteFaceResponse(BaseModel):
    """Delete face response model."""
    success: bool
    user_id: str
    deleted_count: int


class FaceUpdateResponse(BaseModel):
    """Face update response model."""
    success: bool
    user_id: str
    message: str
    previous_embedding_deleted: bool

class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    message: str

class MultiFaceIdentifyItem(BaseModel):
    """Individual face identification result for multi-face detection."""
    face_index: int
    match: bool
    user_id: Optional[str]
    similarity: float
    confidence: float
    bounding_box: Dict[str, int]  # x, y, width, height

class MultiFaceIdentifyResponse(BaseModel):
    """Multi-face identification response model."""
    total_faces_detected: int
    identified_faces: List[MultiFaceIdentifyItem]
    camera_id: Optional[str]
    flag: Optional[str]

# Gateway authentication helper
def verify_gateway_auth(x_authenticated_by: Optional[str]) -> None:
    """
    Verify that the request came through the trusted Nginx gateway.

    In production, this must be "nginx-gateway".
    For development/testing, can be disabled with environment variable.
    """
    require_gateway = os.getenv("REQUIRE_GATEWAY_AUTH", "true").lower() == "true"

    if require_gateway and x_authenticated_by != "nginx-gateway":
        logger.warning(f"Unauthorized request attempt: X-Authenticated-By={x_authenticated_by}")
        raise HTTPException(
            status_code=401,
            detail="Unauthorized - must go through trusted gateway"
        )


# Endpoints

@app.get("/health", response_model=HealthResponse, tags=["Public"])
async def health_check():
    """
    Public health check endpoint.
    Returns service status and Milvus connection status.
    """
    try:
        milvus_client = get_milvus_client()
        stats = milvus_client.get_collection_stats()
        return HealthResponse(
            status="healthy",
            service="Prayas Face Recognition",
            milvus_connected=True,
            collection_stats=stats
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            service="Prayas Face Recognition",
            milvus_connected=False
        )

@app.get("/api/face/health", response_model=HealthResponse, tags=["Public"])
async def health_check_api():
    """
    Public health check endpoint under /api/face/ path.
    Returns service status and Milvus connection status.
    This allows Nginx gateway to check health at /api/face/health
    """
    return await health_check()

@app.post("/api/face/enroll", response_model=FaceEnrollResponse, tags=["Protected"])
async def enroll_face(
    file: UploadFile = File(..., description="Image file containing face to enroll"),
    user_id: str = Form(None),
    x_username: str = Header(None, alias="X-Username"),
    x_authenticated_by: str = Header(None, alias="X-Authenticated-By")
):
    """
    Enroll a face by extracting and storing its embedding in Milvus.

    Protected endpoint - requires JWT via Nginx gateway (X-Authenticated-By header).
    The gateway validates the JWT and injects the X-Authenticated-By header.

    Args:
        file: Image file containing the face to enroll
        user_id: User identifier to associate with this face
        x_username: Organization name (passed via X-Username header from JWT)
        x_authenticated_by: Gateway authentication header (injected by Nginx)

    Returns:
        FaceEnrollResponse with enrollment status
    """
    # Verify gateway authentication
    verify_gateway_auth(x_authenticated_by)

    try:
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")

        # Extract organization name from JWT (passed via X-Username header)
        organization_name = x_username
        if not organization_name:
            raise HTTPException(status_code=400, detail="Organization name not found in token")

        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        logger.info(f"Enrolling face for user: {user_id}, organization: {organization_name}")

        # Extract face embedding from uploaded image
        image_bytes = await file.read()
        embedding, bbox = extract_embedding(image_bytes)

        milvus_client = get_milvus_client()

        # VALIDATION 1: Check if user_id already exists in this organization
        if milvus_client.user_exists(user_id, organization_name):
            logger.warning(f"Enrollment rejected: user_id {user_id} already exists in organization {organization_name}")
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "USER_ALREADY_EXISTS",
                    "message": f"User '{user_id}' already has a face registered in this organization. Use PUT /api/face/update to change.",
                    "user_id": user_id
                }
            )

        # VALIDATION 2: Check if this face is already enrolled for another user in this organization
        existing_face = milvus_client.check_face_duplicate(
            embedding=embedding,
            organization_name=organization_name,
            threshold=THRESHOLD
        )
        if existing_face:
            existing_user_id, similarity = existing_face
            logger.warning(
                f"Enrollment rejected: face already enrolled for user {existing_user_id} "
                f"in organization {organization_name} (similarity: {similarity:.3f})"
            )
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "FACE_ALREADY_ENROLLED",
                    "message": f"This face is already registered for user '{existing_user_id}'",
                    "existing_user_id": existing_user_id,
                    "similarity": float(similarity)
                }
            )

        # All validations passed - store embedding
        entity_id = milvus_client.insert_embedding(user_id, embedding, organization_name)
        logger.info(f"Face enrolled successfully: user={user_id}, organization={organization_name}, entity_id={entity_id}")

        return FaceEnrollResponse(
            success=True,
            user_id=user_id,
            message="Face enrolled successfully"
        )

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Enrollment validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Enrollment error: {e}")
        raise HTTPException(status_code=500, detail=f"Enrollment failed: {str(e)}")


@app.post("/api/face/identify", response_model=FaceIdentifyResponse, tags=["Protected"])
async def identify_face(
    file: UploadFile = File(..., description="Image file containing face to identify"),
    x_username: str = Header(None, alias="X-Username"),
    x_authenticated_by: str = Header(None, alias="X-Authenticated-By")
):
    """
    Identify a face by searching in Milvus within the user's organization.

    Protected endpoint - requires JWT via Nginx gateway.

    Args:
        file: Image file containing the face to identify
        x_username: Organization name (passed via X-Username header from JWT)
        x_authenticated_by: Gateway authentication header (injected by Nginx)

    Returns:
        FaceIdentifyResponse with match status and user ID if found
    """
    # Verify gateway authentication
    verify_gateway_auth(x_authenticated_by)

    try:
        # Extract organization name from JWT (passed via X-Username header)
        organization_name = x_username
        if not organization_name:
            raise HTTPException(status_code=400, detail="Organization name not found in token")

        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        logger.info(f"Processing face identification request for organization: {organization_name}")

        # Extract face embedding from uploaded image
        image_bytes = await file.read()
        test_embedding, _ = extract_embedding(image_bytes)

        # Search in Milvus within the organization
        milvus_client = get_milvus_client()
        matches = milvus_client.search_similar(test_embedding, organization_name, threshold=THRESHOLD)

        if matches:
            best_user_id, best_score = matches[0]
            is_match = True
            logger.info(f"Face identified: {best_user_id} (score: {best_score:.3f})")
        else:
            best_user_id = None
            best_score = 0.0
            is_match = False
            logger.info("No matching face found")

        return FaceIdentifyResponse(
            match=is_match,
            user_id=best_user_id,
            similarity=float(best_score),
            threshold=float(THRESHOLD)
        )

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Identification validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Identification error: {e}")
        raise HTTPException(status_code=500, detail=f"Identification failed: {str(e)}")


@app.put("/api/face/update", response_model=FaceUpdateResponse, tags=["Protected"])
async def update_face(
    file: UploadFile = File(..., description="Image file containing new face"),
    user_id: str = Form(None),
    x_username: str = Header(None, alias="X-Username"),
    x_authenticated_by: str = Header(None, alias="X-Authenticated-By")
):
    """
    Update face embedding for an existing user.

    Protected endpoint - requires JWT via Nginx gateway.

    Allows a legitimate user to update their face embedding.
    Validates that the new face is not already registered for another user.

    Args:
        file: Image file containing the new face
        user_id: User identifier to update
        x_username: Organization name (passed via X-Username header from JWT)
        x_authenticated_by: Gateway authentication header (injected by Nginx)

    Returns:
        FaceUpdateResponse with update status
    """
    # Verify gateway authentication
    verify_gateway_auth(x_authenticated_by)

    try:
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")

        # Extract organization name from JWT (passed via X-Username header)
        organization_name = x_username
        if not organization_name:
            raise HTTPException(status_code=400, detail="Organization name not found in token")

        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        logger.info(f"Updating face for user: {user_id}, organization: {organization_name}")

        # Extract face embedding from uploaded image
        image_bytes = await file.read()
        embedding, bbox = extract_embedding(image_bytes)

        milvus_client = get_milvus_client()

        # Check if user exists
        user_exists = milvus_client.user_exists(user_id, organization_name)

        # Check if new face is already enrolled for another user in same organization
        existing_face = milvus_client.check_face_duplicate(
            embedding=embedding,
            organization_name=organization_name,
            threshold=THRESHOLD,
            exclude_user_id=user_id  # Exclude current user from check
        )

        if existing_face:
            existing_user_id, similarity = existing_face
            logger.warning(
                f"Update rejected: face already enrolled for user {existing_user_id} "
                f"in organization {organization_name} (similarity: {similarity:.3f})"
            )
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "FACE_ALREADY_ENROLLED",
                    "message": f"This face is already registered for user '{existing_user_id}'",
                    "existing_user_id": existing_user_id,
                    "similarity": float(similarity)
                }
            )

        # Update the embedding (delete old, insert new)
        entity_id = milvus_client.update_embedding(user_id, embedding, organization_name)
        logger.info(f"Face updated successfully: user={user_id}, organization={organization_name}, entity_id={entity_id}")

        return FaceUpdateResponse(
            success=True,
            user_id=user_id,
            message="Face updated successfully",
            previous_embedding_deleted=user_exists
        )

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Update validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Update error: {e}")
        raise HTTPException(status_code=500, detail=f"Update failed: {str(e)}")


@app.delete("/api/face/{user_id}", response_model=DeleteFaceResponse, tags=["Protected"])
async def delete_face(
    user_id: str,
    x_username: str = Header(None, alias="X-Username"),
    x_authenticated_by: str = Header(None, alias="X-Authenticated-By")
):
    """
    Delete face embedding for a user within an organization.

    Protected endpoint - requires JWT via Nginx gateway.

    Args:
        user_id: User identifier whose face to delete
        x_username: Organization name (passed via X-Username header from JWT)
        x_authenticated_by: Gateway authentication header (injected by Nginx)

    Returns:
        DeleteFaceResponse with deletion status
    """
    # Verify gateway authentication
    verify_gateway_auth(x_authenticated_by)

    try:
        # Extract organization name from JWT (passed via X-Username header)
        organization_name = x_username
        if not organization_name:
            raise HTTPException(status_code=400, detail="Organization name not found in token")

        milvus_client = get_milvus_client()
        deleted_count = milvus_client.delete_user(user_id, organization_name)

        logger.info(f"Deleted {deleted_count} embeddings for user {user_id} in organization {organization_name}")

        return DeleteFaceResponse(
            success=True,
            user_id=user_id,
            deleted_count=deleted_count
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete error: {e}")
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")


@app.get("/api/face/stats", response_model=CollectionStatsResponse, tags=["Protected"])
async def get_collection_stats(
    x_authenticated_by: str = Header(None, alias="X-Authenticated-By")
):
    """
    Get collection statistics.

    Protected endpoint - requires JWT via Nginx gateway.

    Returns:
        CollectionStatsResponse with collection info
    """
    # Verify gateway authentication
    verify_gateway_auth(x_authenticated_by)

    try:
        milvus_client = get_milvus_client()
        stats = milvus_client.get_collection_stats()
        return CollectionStatsResponse(**stats)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@app.post("/api/face/identify-multiple", response_model=MultiFaceIdentifyResponse, tags=["Protected"])
async def identify_multiple_faces(
    file: UploadFile = File(..., description="Image file containing multiple faces to identify"),
    camera_id: Optional[str] = Form(None),
    flag: Optional[str] = Form(None),
    x_username: str = Header(None, alias="X-Username"),
    x_authenticated_by: str = Header(None, alias="X-Authenticated-By")
):
    """
    Identify multiple faces in a single image by detecting all faces and searching in Milvus.

    Protected endpoint - requires JWT via Nginx gateway.

    This endpoint:
    1. Detects all faces in the uploaded image
    2. Extracts embeddings for each detected face
    3. Searches for matches in Milvus within the organization
    4. Returns list of identified faces with their bounding boxes and similarity scores

    Args:
        file: Image file containing multiple faces to identify
        camera_id: Optional camera identifier for tracking which camera captured the image
        flag: Optional flag for additional context or metadata
        x_username: Organization name (passed via X-Username header from JWT)
        x_authenticated_by: Gateway authentication header (injected by Nginx)

    Returns:
        MultiFaceIdentifyResponse with list of identified faces
    """
    # Verify gateway authentication
    verify_gateway_auth(x_authenticated_by)

    try:
        # Extract organization name from JWT (passed via X-Username header)
        organization_name = x_username
        if not organization_name:
            raise HTTPException(status_code=400, detail="Organization name not found in token")

        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        logger.info(f"Processing multi-face identification request for organization: {organization_name}, camera: {camera_id}, flag: {flag}")

        # Read image bytes
        image_bytes = await file.read()
        
        # Extract multiple faces from the image
        face_detections = extract_multiple_faces(image_bytes)
        
        if not face_detections:
            logger.info("No faces detected in the image")
            return MultiFaceIdentifyResponse(
                total_faces_detected=0,
                identified_faces=[],
                camera_id=camera_id,
                flag=flag
            )

        logger.info(f"Detected {len(face_detections)} faces in the image")

        milvus_client = get_milvus_client()
        identified_faces = []

        db: Session = SessionLocal()

        # Process each detected face
        for idx, detection in enumerate(face_detections):
            embedding = detection["embedding"]
            bbox = detection["bbox"]
            confidence = detection["confidence"]

            # Search for match in Milvus
            matches = milvus_client.search_similar(embedding, organization_name, threshold=THRESHOLD)

            if matches:
                best_user_id, best_score = matches[0]
                identified_faces.append(MultiFaceIdentifyItem(
                    face_index=idx,
                    match=True,
                    user_id=best_user_id,
                    similarity=float(best_score),
                    confidence=float(confidence),
                    bounding_box={
                        "x": int(bbox[0]),
                        "y": int(bbox[1]),
                        "width": int(bbox[2] - bbox[0]),
                        "height": int(bbox[3] - bbox[1])
                    }
                ))
                if best_score >= THRESHOLD:
                    attendance = StudentAttendance(
                        organization_id=organization_name,
                        student_id=best_user_id,
                        camera_id=camera_id,
                        attendance_type=AttendanceType.lecture,   # example enum value
                        event_time=datetime.utcnow(),
                        date=date.today(),
                        confidence_score=float(best_score),
                        created_at=datetime.utcnow()
                    )
                    db.add(attendance)
                    db.commit()
                logger.info(f"Face {idx} identified as {best_user_id} with camera id is : {camera_id} and (score: {best_score:.3f})")
            else:
                # No match found for this face
                identified_faces.append(MultiFaceIdentifyItem(
                    face_index=idx,
                    match=False,
                    user_id=None,
                    similarity=0.0,
                    confidence=float(confidence),
                    bounding_box={
                        "x": int(bbox[0]),
                        "y": int(bbox[1]),
                        "width": int(bbox[2] - bbox[0]),
                        "height": int(bbox[3] - bbox[1])
                    }
                ))
                logger.info(f"Face {idx} not identified")

        return MultiFaceIdentifyResponse(
            total_faces_detected=len(face_detections),
            identified_faces=identified_faces,
            camera_id=camera_id,
            flag=flag
        )

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Multi-face identification validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Multi-face identification error: {e}")
        raise HTTPException(status_code=500, detail=f"Multi-face identification failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT)
