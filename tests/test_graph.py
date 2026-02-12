import unittest
from unittest.mock import MagicMock, patch
from src.state import CuratorState
from src.graph import search_for_content, manage_topics, run_grader

class TestGraphNodes(unittest.TestCase):
    
    @patch('src.graph.tavily_client')
    @patch('src.graph.llm')
    def test_search_for_content_weighted(self, mock_llm, mock_tavily):
        # Setup State
        state = {
            "search_results": [], # Empty to trigger search
            "topic_weights": {"Space": 10}, # Dominant topic
            "topics": ["Space"],
            "user_profile": "Test Profile",
            "viewed_ids": [],
            "search_query": "",
            "loop_count": 0
        }
        
        # Mock LLM to return a query string
        mock_llm.invoke.return_value.content = "Space News Query"
        
        # Mock Tavily Search
        mock_tavily.search.return_value = {"results": [{"url": "http://new.com", "title": "New"}]}
        
        # Run Node
        result = search_for_content(state)
        
        # Assertions
        self.assertIn("search_results", result)
        self.assertEqual(len(result["search_results"]), 1)
        self.assertEqual(result["search_results"][0]["url"], "http://new.com")
        self.assertEqual(result["search_query"], "Space News Query")
        
        # Verify LLM was called to generate specific query
        mock_llm.invoke.assert_called_once()
    
    @patch('src.graph.llm')
    def test_manage_topics_extraction(self, mock_llm):
        # Setup State with one LIKED article
        state = {
            "feedback_history": [{
                "article": {"title": "Mars Mission", "content": "SpaceX goes to Mars"},
                "liked": True
            }],
            "topic_weights": {},
            "topics": []
        }
        
        # Mock LLM response for extraction
        mock_llm.invoke.return_value.content = '["SpaceX", "Mars"]'
        
        result = manage_topics(state)
        
        weights = result["topic_weights"]
        self.assertIn("SpaceX", weights)
        self.assertIn("Mars", weights)
        self.assertEqual(weights["SpaceX"], 1)

if __name__ == '__main__':
    unittest.main()
