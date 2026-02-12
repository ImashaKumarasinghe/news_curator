import os
import json
import numpy as np
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional

# Constants
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "news_articles"

class VectorStore:
    def __init__(self):
        # Initialize Client (Persistent)
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        
        # Setup Embedding Function
        # We try to use OpenAI if available, otherwise fallback to simple default
        if os.environ.get("OPENAI_API_KEY"):
            self.emb_fn = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.environ.get("OPENAI_API_KEY"),
                model_name="text-embedding-3-small"
            )
        else:
            # Fallback for Google or no key (Chroma default uses ONNX MiniLM locally)
            self.emb_fn = embedding_functions.DefaultEmbeddingFunction()

        # Get or Create Collection
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.emb_fn,
            metadata={"hnsw:space": "cosine"} # Use Cosine similarity
        )

    def add_interaction(self, article: Dict, liked: bool):
        """
        Stores an article with its interaction feedback.
        """
        url = article.get("url")
        if not url:
            return

        # Prepare Metadata (Flat structure for filtering)
        # Chroma metadata must be int, float, str, or bool. No nested dicts.
        metadata = {
            "url": url,
            "title": article.get("title", "")[:100], # Truncate for safety
            "liked": liked,
            "timestamp": getattr(article, "timestamp", ""), # If available
            "source_json": json.dumps(article) # Store full data as string
        }

        # Content to Embed: Title + Summary
        content = f"{article.get('title', '')}\n{article.get('content', '')}"

        # Upsert
        self.collection.upsert(
            ids=[url],
            documents=[content],
            metadatas=[metadata]
        )

    def get_preferences_centroid(self) -> Optional[List[float]]:
        """
        Calculates the 'User Centroid' vector:
        Average(Liked Vectors) - 0.2 * Average(Disliked Vectors)
        Returns the raw vector or None if no history.
        """
        # Fetch all Liked
        liked_results = self.collection.get(
            where={"liked": True},
            include=["embeddings"]
        )
        
        # Check if empty (handles None, [], or empty np.array safely)
        liked_emb = liked_results.get("embeddings")
        if liked_emb is None or len(liked_emb) == 0:
            return None
            
        liked_vectors = np.array(liked_emb)
        liked_center = np.mean(liked_vectors, axis=0)

        # Fetch Disliked (Optional: to steer away)
        disliked_results = self.collection.get(
            where={"liked": False},
            include=["embeddings"]
        )

        disliked_emb = disliked_results.get("embeddings")
        if disliked_emb is not None and len(disliked_emb) > 0:
            disliked_vectors = np.array(disliked_emb)
            disliked_center = np.mean(disliked_vectors, axis=0)
            
            # Weighted Centroid: Focus on liked, push slightly away from disliked
            # Formula: Target = Liked_Center - (0.2 * Disliked_Center)
            # We normalize afterwards implicitly by Cosine Similarity usage later
            centroid = liked_center - (0.2 * disliked_center)
            return centroid.tolist()
            
        return liked_center.tolist()

    def get_all_viewed_ids(self) -> List[str]:
        """Returns list of all article IDs (URLs) stored in DB."""
        # Simple fetch of all IDs
        results = self.collection.get(include=[])
        return results["ids"]

    def rerank_results(self, candidates: List[Dict]) -> List[Dict]:
        """
        Takes a list of search results, embeds them locally, 
        and ranks them by similarity to the user centroid.
        """
        user_centroid = self.get_preferences_centroid()
        if user_centroid is None:
            return candidates # No history, return as is (Tavily rank)

        # We can't query the DB because these candidates aren't IN the DB yet.
        # We must embed them on the fly.
        
        texts = [f"{c.get('title', '')}\n{c.get('content', '')}" for c in candidates]
        
        # Generate embeddings for candidates
        candidate_embeddings = self.emb_fn(texts)
        
        # Calculate Cosine Similarity manually
        # Sim(A, B) = dot(A, B) / (norm(A) * norm(B))
        
        centroid_vec = np.array(user_centroid)
        centroid_norm = np.linalg.norm(centroid_vec)
        
        scored_candidates = []
        for i, emb in enumerate(candidate_embeddings):
            vec = np.array(emb)
            norm = np.linalg.norm(vec)
            if norm == 0 or centroid_norm == 0:
                score = 0
            else:
                score = np.dot(vec, centroid_vec) / (norm * centroid_norm)
            
            scored_candidates.append({
                "article": candidates[i],
                "score": score
            })
            
        # Sort desc
        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        
        # Return strict list of articles
        return [item["article"] for item in scored_candidates]

    def close(self):
        # Chroma persistent client handles itself usually, but good practice stub
        pass
