def test_mcp_server_module_imports_and_registers_tools():
    from processmind.mcp.server import get_process_spec_schema, mcp_server, search_knowledge_base

    assert mcp_server.name == "processmind-mcp"
    assert callable(search_knowledge_base)
    assert callable(get_process_spec_schema)


def test_get_process_spec_schema_returns_schema():
    from processmind.mcp.server import get_process_spec_schema

    schema = get_process_spec_schema()
    assert schema["title"] == "ProcessSpec"
    assert "steps" in schema["properties"]
