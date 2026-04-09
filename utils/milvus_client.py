"""
Milvus client for face embedding storage and search.
Handles vector database operations for the face recognition service.
"""

from pymilvus import (
    connections, Collection, FieldSchema, CollectionSchema, DataType, utility
)
import numpy as np
from typing import List, Tuple, Optional
import logging

from .face_config import MILVUS_HOST, MILVUS_PORT, COLLECTION_NAME, EMBEDDING_DIM

logger = logging.getLogger(__name__)


class MilvusClient:
    """Client for Milvus vector database operations."""

    def __init__(self):
        """Initialize Milvus connection and collection."""
        self._connect()
        self.collection = self._get_or_create_collection()

    def _connect(self):
        """Establish connection to Milvus server."""
        try:
            connections.connect(
                alias="default",
                host=MILVUS_HOST,
                port=MILVUS_PORT
            )
            logger.info(f"Connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise

    def _get_or_create_collection(self) -> Collection:
        """Get existing collection or create new one."""
        if utility.has_collection(COLLECTION_NAME):
            logger.info(f"Using existing collection: {COLLECTION_NAME}")
            collection = Collection(COLLECTION_NAME)
            collection.load(_resource_timeout=5)
            return collection

        # Define schema for face embeddings
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="organization_name", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
            FieldSchema(name="created_at", dtype=DataType.INT64),
        ]

        schema = CollectionSchema(fields=fields, description="Face embeddings for attendance system")
        collection = Collection(name=COLLECTION_NAME, schema=schema)

        # Create index with IVF_FLAT for efficient cosine similarity search
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 128}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        collection.load(_resource_timeout=5)

        logger.info(f"Created new collection: {COLLECTION_NAME}")
        return collection

    def insert_embedding(self, user_id: str, embedding: np.ndarray, organization_name: str) -> int:
        """
        Insert face embedding for a user.

        Args:
            user_id: User identifier
            embedding: Normalized face embedding vector
            organization_name: Organization name for multi-tenant isolation

        Returns:
            The ID of the inserted entity
        """
        data = [{
            "user_id": user_id,
            "organization_name": organization_name,
            "embedding": embedding.tolist(),
            "created_at": int(__import__("time").time() * 1000)
        }]

        insert_result = self.collection.insert(data)
        #self.collection.flush()
        logger.info(f"Inserted embedding for user {user_id}, entity ID: {insert_result.primary_keys[0]}")
        return insert_result.primary_keys[0]

    def search_similar(
        self,
        embedding: np.ndarray,
        organization_name: str,
        threshold: float = 0.5,
        top_k: int = 1
    ) -> List[Tuple[str, float]]:
        """
        Search for similar faces in the collection within an organization.

        Args:
            embedding: Query face embedding vector
            organization_name: Organization name to filter search scope
            threshold: Minimum similarity score (0-1) for a match
            top_k: Maximum number of results to return

        Returns:
            List of (user_id, score) tuples for matches above threshold
        """
        results = self.collection.search(
            data=[embedding.tolist()],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 16}},
            limit=top_k,
            expr=f'organization_name == "{organization_name}"',
            output_fields=["user_id"]
        )

        matches = []
        for hit in results[0]:
            if hit.score >= threshold:
                matches.append((hit.entity.get("user_id"), hit.score))
                logger.info(f"Match found: {hit.entity.get('user_id')} with score {hit.score:.3f}")

        if not matches:
            logger.info(f"No matches found above threshold {threshold}")

        return matches

    def delete_user(self, user_id: str, organization_name: str) -> int:
        """
        Delete all embeddings for a specific user within an organization.

        Args:
            user_id: User identifier
            organization_name: Organization name to scope the deletion

        Returns:
            Number of embeddings deleted
        """
        # Query to find all embeddings for this user in this organization
        results = self.collection.query(
            expr=f'user_id == "{user_id}" && organization_name == "{organization_name}"',
            output_fields=["id", "user_id"]
        )

        if not results:
            logger.info(f"No embeddings found for user {user_id}")
            return 0

        logger.info(f"Found {len(results)} embeddings for user {user_id}: {results}")

        # Extract primary keys - Milvus returns primary key as 'id' field
        ids_to_delete = [r.get("id") for r in results if r.get("id") is not None]
        if ids_to_delete:
            self.collection.delete(expr=f'id in {ids_to_delete}')
            logger.info(f"Deleted {len(ids_to_delete)} embeddings for user {user_id}")
        return len(ids_to_delete)

    def user_exists(self, user_id: str, organization_name: str) -> bool:
        """
        Check if a user_id already has a face registered in an organization.

        Args:
            user_id: User identifier to check
            organization_name: Organization name to scope the check

        Returns:
            True if user_id exists in collection for this organization, False otherwise
        """
        results = self.collection.query(
            expr=f'user_id == "{user_id}" && organization_name == "{organization_name}"',
            output_fields=["user_id"],
            limit=1
        )
        return len(results) > 0

    def check_face_duplicate(
        self,
        embedding: np.ndarray,
        organization_name: str,
        threshold: float = 0.5,
        exclude_user_id: str = None
    ) -> Optional[Tuple[str, float]]:
        """
        Check if a similar face already exists in the collection within an organization.

        Args:
            embedding: Face embedding vector to check
            organization_name: Organization name to scope the search
            threshold: Similarity threshold for considering faces as duplicate
            exclude_user_id: Optional user_id to exclude from check (for updates)

        Returns:
            Tuple of (existing_user_id, similarity_score) if duplicate found, None otherwise
        """
        results = self.collection.search(
            data=[embedding.tolist()],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 16}},
            limit=1,
            expr=f'organization_name == "{organization_name}"',
            output_fields=["user_id"]
        )

        if results and results[0]:
            hit = results[0][0]
            if hit.score >= threshold:
                existing_user_id = hit.entity.get("user_id")
                # Exclude the same user if specified (for update scenarios)
                if exclude_user_id and existing_user_id == exclude_user_id:
                    return None
                return (existing_user_id, hit.score)

        return None

    def update_embedding(self, user_id: str, embedding: np.ndarray, organization_name: str) -> int:
        """
        Update face embedding for an existing user.
        Deletes old embedding(s) and inserts new one.

        Args:
            user_id: User identifier
            embedding: New normalized face embedding vector
            organization_name: Organization name for multi-tenant isolation

        Returns:
            The ID of the newly inserted entity
        """
        # Delete existing embeddings for this user in this organization
        self.delete_user(user_id, organization_name)

        # Insert new embedding
        return self.insert_embedding(user_id, embedding, organization_name)

    def get_collection_stats(self) -> dict:
        """Get collection statistics."""
        return {
            "num_entities": self.collection.num_entities,
            "name": self.collection.name
        }

    def close(self):
        """Close Milvus connection."""
        connections.disconnect("default")
        logger.info("Disconnected from Milvus")


# Singleton instance
_milvus_client: Optional[MilvusClient] = None


def get_milvus_client() -> MilvusClient:
    """Get or create Milvus client singleton."""
    global _milvus_client
    if _milvus_client is None:
        _milvus_client = MilvusClient()
    return _milvus_client