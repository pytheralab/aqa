import os
import logging
import json

from typing import Optional, List, Any
from qdrant_client.http import models
from qdrant_client import QdrantClient

from .interface import InterfaceDatabase

DISTANCE_MAPPING = {
    'euclidean': models.Distance.EUCLID,
    'dot': models.Distance.DOT, 
    'manhattan': models.Distance.MANHATTAN,
    'cosine': models.Distance.COSINE
}


class QdrantChunksDB(InterfaceDatabase):
    """ 
    Vector database using Qdrant for storing and searching chunks and documents.
    """
    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self._client = self.connect_client(url, api_key)

    def connect_client(self, url, api_key):
        if url is not None and api_key is not None:
            # cloud instance
            return QdrantClient(url=url, api_key=api_key)
        elif url is not None:
            # local instance with differ url
            return QdrantClient(url=url)
        else:
            logging.error("None client connection")
            return None
        
    def create_colection(self, chunker_id="db", dimension=768, distance='cosine') -> None:
        # resolve distance
        distance = DISTANCE_MAPPING.get(distance, models.Distance.COSINE)
        self._client.create_collection(
            collection_name=chunker_id,
            vectors_config=models.VectorParams(
                size=dimension, distance=distance
            ),
        )

    def insert(self, chunks: dict, chunker_id: str, **kwargs) -> None:
        """ Insert points into collection """

        # Check if chunkers exists
        if not self._client.collection_exists(chunker_id):
            dim = kwargs.get('dimension', len(chunks['vectors'][0])) # specify dim or auto detect
            distance = kwargs.get('distance', 'cosine')
            self.create_colection(chunker_id, dimension=dim, distance=distance)
        else:
            logging.info(f"Collection {chunker_id} already exists")
        # Insert chunks
        self._client.upload_collection(
            collection_name=chunker_id,
            ids=chunks['ids'],
            payload=chunks['payloads'],
            vectors=chunks['vectors'],
            # ids of chunks are not provided, Qdrant Client will generate them automatically as random UUIDs.
        )

    def search(self, chunk_emb: List[float], chunker_id: str, top_k:int = 5):
        # Top-k passages
        chunks = self._client.search(
            collection_name=chunker_id,
            query_vector=chunk_emb,
            limit=top_k,
        )
        chunks = [{
            "id": chunk.id,
            "score": chunk.score,
            "payload": chunk.payload
        } for chunk in chunks]
        return chunks
    
    def get_chunks(self, chunker_id: str, limit: int = 5):
        """ Get all chunks from collection """
        results = self._client.scroll(
            collection_name=chunker_id,
            limit=limit,
        )
        # Check if results are empty
        if not results[0]:
            return []
        # Format results
        results = [
            {
                "id": chunk.id,
                "payload": chunk.payload,
            }
            for chunk in results[0]
        ]
        return results

    def get_chunks_by_doc_id(self, doc_id: str, chunker_id: str):
        """ Get document information using doc_id """
        results =  self._client.scroll(
            collection_name=chunker_id,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_id", match=models.MatchValue(value=doc_id)
                    ),
                ],
            ),
        )
        # Check if results are empty
        if not results[0]:
            return []
        # Format results
        results = [
            {
                "id": chunk.id,
                "payload": chunk.payload,
            }
            for chunk in results[0]
        ]
        return results
    
    def delete(self, chunk_ids: str | List[str] | None, doc_id: str | None, chunker_id: str, **kwagrs):
        try:
            if chunk_ids is not None:
                if isinstance(chunk_ids, str):
                    chunk_ids = [chunk_ids]
                self._client.delete(
                    collection_name=chunker_id,
                    points_selector=models.PointIdsList(
                        points=chunk_ids,
                    ),
                )
                return {'status': 'success', 'message': f"Chunks [{chunk_ids}] deleted!"}
            elif doc_id is not None:
                self._client.delete(
                    collection_name=chunker_id,
                    points_selector=models.FilterSelector(filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="doc_id",
                                match=models.MatchValue(value=f"{doc_id}"),
                            ),
                        ])
                    ),
                )
                return {'status': 'success', 'message': f"Document {doc_id} deleted!"}
            else:
                return {'status': 'failed', 'message': 'No group id found'}
            
        except Exception as e:
            return json.loads(e.content.decode('utf-8')) # qdrant error message
    
    def delete_chunker(self, chunker_id: str):
        try:
            self._client.delete_collection(collection_name=chunker_id)
            return {'status': 'success', 'message': f"Collection {chunker_id} deleted!"}
        except Exception as e:
            return {'status': 'failed', 'message': f"Error deleting collection {chunker_id}: {e}"}
        
    def update(self, points_ids, **kwagrs):
        ...