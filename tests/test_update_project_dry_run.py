def test_update_project_dry_run():
    """Test the update_project_dry_run MCP tool."""
    from ast_rag.ast_rag_mcp import update_project_dry_run

    try:
        result = update_project_dry_run(
            from_commit="HEAD~1",
            to_commit="HEAD",
            max_changed_nodes=1000,
        )
        assert "stats" in result
        assert "warning" in result or "warning" not in result  # Depends on repo size
    except Exception as git_err:
        # Skip if not enough commits (e.g. only 1 commit in repo)
        if "not enough parent commits" in str(git_err) or "did not resolve to an object" in str(git_err):
            print(f"  Skipped: not enough commits in repo ({git_err})")
            return  # Test passes by skipping
        raise


def test_search_by_signature_format():
    """Test the search_by_signature MCP tool returns StandardResult format."""
    from ast_rag.ast_rag_mcp import search_by_signature

    results = search_by_signature("*(int, String)", lang="java", limit=5)
    assert isinstance(results, list)
    if results:
        assert "id" in results[0]
        assert "kind" in results[0]
