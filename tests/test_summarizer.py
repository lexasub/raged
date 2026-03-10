"""
tests/test_summarizer.py - Tests for the code summarization service.

Run with: pytest tests/test_summarizer.py -v
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from ast_rag.summarizer import (
    SummarizerService,
    NodeSummary,
    ComplexityLevel,
    SummaryCacheEntry,
    SUMMARY_PROMPT_TEMPLATE,
)
from ast_rag.models import ASTNode, NodeKind, Language


class TestNodeSummary:
    """Tests for NodeSummary model."""

    def test_create_minimal_summary(self):
        """Test creating a minimal NodeSummary."""
        summary = NodeSummary(
            node_id="test123",
            summary="Test function summary",
        )
        
        assert summary.node_id == "test123"
        assert summary.summary == "Test function summary"
        assert summary.inputs == []
        assert summary.outputs == []
        assert summary.side_effects == []
        assert summary.calls == []
        assert summary.called_by == []
        assert summary.complexity == ComplexityLevel.MEDIUM
        assert summary.tags == []
        assert summary.model_used is None

    def test_create_full_summary(self):
        """Test creating a complete NodeSummary."""
        summary = NodeSummary(
            node_id="abc456",
            summary="Processes user requests",
            inputs=[
                {"name": "user_id", "type": "str", "description": "User identifier"},
                {"name": "data", "type": "dict", "description": "Request payload"},
            ],
            outputs=[
                {"name": "return", "type": "Response", "description": "Processed response"},
            ],
            side_effects=[
                "Database write operation",
                "Logs user activity",
            ],
            calls=[
                "com.example.db.UserRepository.save",
                "com.example.logger.InfoLogger.log",
            ],
            called_by=[
                "com.example.controller.UserController.handle",
            ],
            complexity=ComplexityLevel.HIGH,
            tags=["async", "io", "validated"],
            model_used="qwen2.5-coder:14b",
        )
        
        assert len(summary.inputs) == 2
        assert len(summary.outputs) == 1
        assert len(summary.side_effects) == 2
        assert len(summary.calls) == 2
        assert len(summary.called_by) == 1
        assert summary.complexity == ComplexityLevel.HIGH
        assert "async" in summary.tags

    def test_to_dict(self):
        """Test converting summary to dictionary."""
        summary = NodeSummary(
            node_id="test789",
            summary="Test summary",
            complexity=ComplexityLevel.LOW,
        )
        
        result = summary.to_dict()
        
        assert isinstance(result, dict)
        assert result["node_id"] == "test789"
        assert result["summary"] == "Test summary"
        assert result["complexity"] == "low"

    def test_to_markdown(self):
        """Test Markdown rendering."""
        summary = NodeSummary(
            node_id="md123",
            summary="Markdown test summary",
            inputs=[{"name": "x", "type": "int", "description": "Input value"}],
            outputs=[{"name": "return", "type": "int", "description": "Result"}],
            side_effects=["None (pure function)"],
            complexity=ComplexityLevel.LOW,
            tags=["pure", "getter"],
        )
        
        md = summary.to_markdown()
        
        assert "## Summary:" in md
        assert "Markdown test summary" in md
        assert "### Inputs" in md
        assert "`x`" in md
        assert "### Outputs" in md
        assert "### Side Effects" in md
        assert "**Complexity:** low" in md
        assert "**Tags:** pure, getter" in md


class TestSummaryPromptTemplate:
    """Tests for the prompt template."""

    def test_prompt_template_structure(self):
        """Test that prompt template has required sections."""
        prompt = SUMMARY_PROMPT_TEMPLATE
        
        assert "Signature" in prompt
        assert "Source Code" in prompt
        assert "Context" in prompt
        assert "Called by This Code" in prompt
        assert "That Call This Code" in prompt
        assert "Instructions" in prompt
        assert "Response Format" in prompt
        assert "JSON" in prompt

    def test_prompt_template_formatting(self):
        """Test formatting the prompt template with data."""
        formatted = SUMMARY_PROMPT_TEMPLATE.format(
            signature="def test_func(x: int) -> str",
            lang="python",
            code="def test_func(x: int) -> str:\n    return str(x)",
            calls_context="- com.example.other_func",
            callers_context="- com.example.caller_func",
        )
        
        assert "def test_func(x: int) -> str" in formatted
        assert "```python" in formatted
        assert "com.example.other_func" in formatted
        assert "com.example.caller_func" in formatted


class TestSummarizerService:
    """Tests for SummarizerService."""

    def test_init_defaults(self):
        """Test default initialization."""
        service = SummarizerService()
        
        assert service._base_url == "http://localhost:11434/v1"
        assert service._model == "qwen2.5-coder:14b"
        assert service._api_key == "ollama"
        assert service._timeout == 120
        assert service._cache_enabled is True

    def test_init_custom(self):
        """Test custom initialization."""
        service = SummarizerService(
            base_url="http://custom:8000/v1",
            model="test-model",
            api_key="test-key",
            timeout=60,
            cache_enabled=False,
        )
        
        assert service._base_url == "http://custom:8000/v1"
        assert service._model == "test-model"
        assert service._api_key == "test-key"
        assert service._timeout == 60
        assert service._cache_enabled is False

    def test_compute_code_hash(self):
        """Test code hash computation."""
        service = SummarizerService()
        
        code1 = "def test(): pass"
        code2 = "def test(): pass"
        code3 = "def other(): pass"
        
        hash1 = service._compute_code_hash(code1)
        hash2 = service._compute_code_hash(code2)
        hash3 = service._compute_code_hash(code3)
        
        assert hash1 == hash2  # Same code = same hash
        assert hash1 != hash3  # Different code = different hash
        assert len(hash1) == 24  # Hash is 24 chars

    def test_cache_operations(self):
        """Test cache save and load."""
        service = SummarizerService(cache_enabled=True)
        
        # Create a test summary
        summary = NodeSummary(
            node_id="cache_test",
            summary="Cached summary",
        )
        
        # Cache it
        code_hash = service._compute_code_hash("test code")
        service._cache_summary("cache_test", code_hash, summary)
        
        # Retrieve from cache
        cached = service._get_cached_summary("cache_test", code_hash)
        
        assert cached is not None
        assert cached.node_id == "cache_test"
        assert cached.summary == "Cached summary"

    def test_cache_invalidates_on_code_change(self):
        """Test that cache is invalidated when code changes."""
        service = SummarizerService(cache_enabled=True)
        
        summary = NodeSummary(
            node_id="invalidate_test",
            summary="Old summary",
        )
        
        # Cache with old code hash
        old_hash = service._compute_code_hash("old code")
        service._cache_summary("invalidate_test", old_hash, summary)
        
        # Try to retrieve with new code hash
        new_hash = service._compute_code_hash("new code")
        cached = service._get_cached_summary("invalidate_test", new_hash)
        
        assert cached is None  # Cache miss due to code change

    def test_parse_llm_response_plain_json(self):
        """Test parsing plain JSON response."""
        service = SummarizerService()
        
        response = '''
        {
            "summary": "Test summary",
            "inputs": [],
            "outputs": [],
            "side_effects": ["None"],
            "calls": [],
            "called_by": [],
            "complexity": "low",
            "tags": []
        }
        '''
        
        parsed = service._parse_llm_response(response)
        
        assert parsed["summary"] == "Test summary"
        assert parsed["complexity"] == "low"

    def test_parse_llm_response_markdown_wrapped(self):
        """Test parsing Markdown-wrapped JSON response."""
        service = SummarizerService()
        
        response = '''
        ```json
        {
            "summary": "Markdown wrapped summary",
            "inputs": [],
            "outputs": [],
            "side_effects": ["None"],
            "calls": [],
            "called_by": [],
            "complexity": "medium",
            "tags": ["test"]
        }
        ```
        '''
        
        parsed = service._parse_llm_response(response)
        
        assert parsed["summary"] == "Markdown wrapped summary"
        assert parsed["complexity"] == "medium"
        assert "test" in parsed["tags"]

    @pytest.mark.asyncio
    async def test_call_llm_async(self):
        """Test async LLM call (mocked)."""
        service = SummarizerService()
        
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"summary": "Mock summary", "inputs": [], "outputs": [], "side_effects": [], "calls": [], "called_by": [], "complexity": "low", "tags": []}'
                    }
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()
        
        with patch.object(service, '_client') as mock_client:
            mock_client.post = AsyncMock(return_value=mock_response)
            
            result = await service._call_llm_async("Test prompt")
            
            assert "Mock summary" in result


class TestComplexityLevel:
    """Tests for ComplexityLevel enum."""

    def test_complexity_values(self):
        """Test complexity level values."""
        assert ComplexityLevel.LOW.value == "low"
        assert ComplexityLevel.MEDIUM.value == "medium"
        assert ComplexityLevel.HIGH.value == "high"

    def test_complexity_from_string(self):
        """Test creating ComplexityLevel from string."""
        assert ComplexityLevel("low") == ComplexityLevel.LOW
        assert ComplexityLevel("medium") == ComplexityLevel.MEDIUM
        assert ComplexityLevel("high") == ComplexityLevel.HIGH


class TestSummaryCacheEntry:
    """Tests for SummaryCacheEntry model."""

    def test_create_cache_entry(self):
        """Test creating a cache entry."""
        summary = NodeSummary(
            node_id="entry_test",
            summary="Test",
        )
        
        entry = SummaryCacheEntry(
            node_id="entry_test",
            code_hash="abc123",
            summary=summary,
        )
        
        assert entry.node_id == "entry_test"
        assert entry.code_hash == "abc123"
        assert entry.summary == summary
        assert entry.created_at is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
