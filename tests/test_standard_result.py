def test_standard_result_creation():
    from ast_rag.models import StandardResult
    
    result = StandardResult(
        id="test123",
        name="processRequest",
        qualified_name="com.example.Handler.processRequest",
        kind="Method",
        lang="java",
        file_path="src/main/java/com/example/Handler.java",
        start_line=42,
        end_line=85,
        score=0.87,
        edge_type="CALLS",
    )
    assert result.id == "test123"
    assert result.score == 0.87
    assert "processRequest" in result.to_markdown()
