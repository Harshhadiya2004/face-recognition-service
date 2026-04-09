"""
Configuration for Prayas face recognition service.
Uses Milvus vector database for embedding storage.
"""

import os

# Milvus Configuration
MILVUS_HOST = os.getenv("MILVUS_HOST", "milvus-standalone")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
COLLECTION_NAME = "face_embeddings"

# Model Configuration
MODEL_NAME = "buffalo_l"
PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
DET_SIZE = (640, 640)
CTX_ID = 0

# Recognition Configuration
EMBEDDING_DIM = 512  # 512-dimensional embeddings from buffalo_l model
THRESHOLD = 0.50     # Cosine similarity threshold for matching

# Service Configuration
SERVICE_PORT = 8001
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
