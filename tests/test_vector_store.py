import unittest
import numpy as np
import shutil
import os
from unittest.mock import MagicMock, patch
from src.vector_store import VectorStore

class TestVectorStore(unittest.TestCase):
    def setUp(self):
        # Mock the PersistentClient to use an in-memory client or a temp dir for tests
        # Since VectorStore hardcodes "chroma_db", we might need to patch the init or the path
        self.test_db_path = "test_chroma_db"
        
        # Patching the path constant if we can, or just patching the client creation
        # easier to patch the class variable or the module constant
        # Python constants in other modules are hard to patch directly if used in __init__ defaults,
        # but here it is used inside __init__.
        
        # We will patch chromadb.PersistentClient
        self.chroma_patcher = patch('src.vector_store.chromadb.PersistentClient')
        self.mock_client_class = self.chroma_patcher.start()
        
        # Setup the mock client and collection
        self.mock_client = MagicMock()
        self.mock_collection = MagicMock()
        self.mock_client_class.return_value = self.mock_client
        self.mock_client.get_or_create_collection.return_value = self.mock_collection
        
        # Create instance
        self.store = VectorStore()
        # Ensure we are using the mock
        self.store.client = self.mock_client
        self.store.collection = self.mock_collection

    def tearDown(self):
        self.chroma_patcher.stop()

    def test_add_interaction(self):
        article = {"url": "http://test.com", "title": "Test", "content": "Content"}
        self.store.add_interaction(article, True)
        
        # Verify upsert was called
        self.mock_collection.upsert.assert_called_once()
        args, kwargs = self.mock_collection.upsert.call_args
        
        self.assertEqual(kwargs['ids'], ["http://test.com"])
        self.assertEqual(kwargs['documents'], ["Test\nContent"])
        self.assertEqual(kwargs['metadatas'][0]['liked'], True)

    def test_get_preferences_centroid_no_data(self):
        # Mock empty return for get()
        self.mock_collection.get.return_value = {"embeddings": []}
        
        centroid = self.store.get_preferences_centroid()
        self.assertIsNone(centroid)

    def test_get_preferences_centroid_calculation(self):
        # Mock embeddings
        # Liked: [1, 1], [3, 3] -> Mean: [2, 2]
        # Disliked: [10, 10]
        # Formula: LikedMean - 0.2 * DislikedMean
        # [2, 2] - 0.2 * [10, 10] = [2, 2] - [2, 2] = [0, 0]
        
        def get_side_effect(where, include):
            if where.get("liked") is True:
                return {"embeddings": [[1, 1], [3, 3]]}
            elif where.get("liked") is False:
                return {"embeddings": [[10, 10]]}
            return {"embeddings": []}

        self.mock_collection.get.side_effect = get_side_effect
        
        centroid = self.store.get_preferences_centroid()
        self.assertTrue(np.allclose(centroid, [0.0, 0.0]), f"Expected [0,0] but got {centroid}")

if __name__ == '__main__':
    unittest.main()
