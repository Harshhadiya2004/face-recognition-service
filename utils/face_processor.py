"""
Core face processing logic for Prayas backend integration.
Handles face detection and embedding extraction using InsightFace.
Vector storage and search are handled by Milvus.
"""

import cv2
import numpy as np
from numpy.linalg import norm
from insightface.app import FaceAnalysis
from typing import List, Dict, Any, Optional, Tuple

from .face_config import MODEL_NAME, PROVIDERS, DET_SIZE, CTX_ID

# Global model instance (singleton pattern)
_face_app = None


def get_face_app():
    """Get or initialize the FaceAnalysis model (singleton)."""
    global _face_app
    if _face_app is None:
        _face_app = FaceAnalysis(name=MODEL_NAME, providers=PROVIDERS)
        _face_app.prepare(ctx_id=CTX_ID, det_size=DET_SIZE)
    return _face_app


def load_image_from_bytes(bytes_data):
    """Load image from bytes data (for file uploads)."""
    nparr = np.frombuffer(bytes_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image from uploaded file")
    return img


def extract_embedding_from_image(img):
    """Extract face embedding from an image array."""
    app = get_face_app()
    faces = app.get(img)
    if len(faces) == 0:
        raise ValueError("No face detected")
    if len(faces) > 1:
        raise ValueError("Multiple faces detected")
    return faces[0].embedding, faces[0].bbox


def extract_embedding(bytes_data):
    """
    Extract embedding from bytes data.
    Returns normalized embedding and bounding box.
    """
    img = load_image_from_bytes(bytes_data)
    embedding, bbox = extract_embedding_from_image(img)
    # Normalize the embedding (unit vector)
    return embedding / norm(embedding), bbox


def cosine_similarity(v1, v2):
    """Calculate cosine similarity between two vectors."""
    return np.dot(v1, v2) / (norm(v1) * norm(v2))

def detect_faces_with_confidence(img: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
    """
    Detect faces in image and return bounding boxes with confidence scores using InsightFace.

    Args:
        img: BGR image as numpy array

    Returns:
        List of tuples (x1, y1, x2, y2, confidence)
    """
    app = get_face_app()
    faces = app.get(img)
    
    results = []
    for face in faces:
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        confidence = face.det_score
        
        results.append((x1, y1, x2, y2, float(confidence)))
    
    return results


def extract_multiple_faces(image_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Extract multiple faces from image bytes using InsightFace.
    Returns list of embeddings and bounding boxes for all detected faces.

    Args:
        image_bytes: Raw image bytes

    Returns:
        List of dictionaries containing:
            - embedding: numpy array of face embedding
            - bbox: tuple of (x1, y1, x2, y2)
            - confidence: detection confidence score
    """
    # Load image from bytes
    img = load_image_from_bytes(image_bytes)
    
    # Get face app
    app = get_face_app()
    
    # Detect all faces
    faces = app.get(img)
    
    if len(faces) == 0:
        return []
    
    results = []
    for face in faces:
        # Get embedding (already normalized by InsightFace)
        embedding = face.embedding
        # Normalize to unit vector (InsightFace embeddings may already be normalized)
        embedding = embedding / norm(embedding)
        
        # Get bounding box
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        
        # Get detection confidence
        confidence = face.det_score
        
        results.append({
            "embedding": embedding,
            "bbox": (x1, y1, x2, y2),
            "confidence": float(confidence)
        })
    
    return results