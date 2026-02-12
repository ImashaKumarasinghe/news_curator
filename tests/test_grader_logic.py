import unittest
from unittest.mock import MagicMock, patch
from src.state import CuratorState
from src.graph import run_grader

class TestGraderLogic(unittest.TestCase):
    
    def test_content_length_filtering(self):
        # Setup State with short content
        state = {
            "search_results": [
                {"title": "Short", "content": "Too short"},
                {"title": "Long", "content": "This is a long enough article content that should pass the length check. " * 10}
            ],
            "vector_store": None,
            "topics": ["Test"],
            "user_profile": "Test Profile"
        }
        
        # Mock grade_documents fallback to just return what's passed
        with patch('src.graph.grade_documents') as mock_grade:
            mock_grade.side_effect = lambda state, llm: {"search_results": state["search_results"], "relevance_score": 1}
            
            result = run_grader(state)
            
            # Should filter out the "Short" article
            self.assertEqual(len(result["search_results"]), 1)
            self.assertEqual(result["search_results"][0]["title"], "Long")

    def test_authoritative_boost(self):
        # Setup results with mixed authority
        # Assuming vector store returns them in semantic order, we check if authority boosts them.
        top_semantic = {"title": "Unknown Source", "url": "http://blog.unknown.com/news", "content": "Content "*20}
        mid_semantic = {"title": "BBC News", "url": "http://bbc.com/news/article", "content": "Content "*20}
        low_semantic = {"title": "Another Unknown", "url": "http://random.com/news", "content": "Content "*20}
        
        results = [top_semantic, mid_semantic, low_semantic]
        
        # Mock Vector Store
        mock_vs = MagicMock()
        # Rerank returns semantic order (simulated)
        mock_vs.rerank_results.return_value = results
        
        state = {
            "search_results": results,
            "vector_store": mock_vs
        }
        
        result = run_grader(state)
        
        # Expect BBC to be boosted to top
        final_list = result["search_results"]
        self.assertEqual(final_list[0]["title"], "BBC News")
        self.assertEqual(final_list[1]["title"], "Unknown Source")
        self.assertEqual(final_list[2]["title"], "Another Unknown")

if __name__ == '__main__':
    unittest.main()
