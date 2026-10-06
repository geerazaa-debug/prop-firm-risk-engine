"""MCP server bootstrap across mcp generations (2.x MCPServer, 1.x FastMCP, standalone fastmcp).

Stdlib-only tests fake the import system; the last test uses the real installed ``mcp``
package when present (skipped otherwise), whichever generation it is.
"""
from __future__ import annotations

import asyncio
import importlib
import importlib.util
import types

import pytest

from mcp_server import mcp_bridge
from mcp_server.mcp_bridge import McpUnavailableError, OrchestratorService


class _FakeServer:
    def __init__(self, name: str) -> None:
        self.name, self.tools = name, {}

    def add_tool(self, fn, name=None, description=None):
        self.tools[name] = (fn, description)


@pytest.fixture
def only_modules(monkeypatch):
    """Make the MCP module names resolve to exactly the given {module: {attr: value}} map;
    every other import (zoneinfo/tzdata, ...) goes to the real import system."""
    real_import, real_find_spec = importlib.import_module, importlib.util.find_spec

    def is_mcp(name: str) -> bool:
        return name.split(".")[0] in ("mcp", "fastmcp")

    def install(available: dict[str, dict[str, object]], installed: tuple[str, ...] = ()):
        def fake_import(name: str, package=None):
            if not is_mcp(name):
                return real_import(name, package)
            if name not in available:
                raise ModuleNotFoundError(f"No module named {name!r}")
            return types.SimpleNamespace(**available[name])

        def fake_find_spec(name: str, package=None):
            if not is_mcp(name):
                return real_find_spec(name, package)
            return object() if name in installed else None

        monkeypatch.setattr(importlib, "import_module", fake_import)
        monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    return install


@pytest.mark.parametrize("module, attr", mcp_bridge.MCP_SERVER_CLASSES)
def test_each_generation_is_supported(only_modules, module, attr):
    marker = type(attr, (_FakeServer,), {})
    only_modules({module: {attr: marker}}, installed=(module.split(".")[0],))
    assert mcp_bridge.load_mcp_server_class() is marker


def test_mcp_2x_is_preferred_over_1x(only_modules):
    v2 = type("MCPServer", (_FakeServer,), {})
    v1 = type("FastMCP", (_FakeServer,), {})
    only_modules({"mcp.server.mcpserver": {"MCPServer": v2}, "mcp.server.fastmcp": {"FastMCP": v1}}, ("mcp",))
    assert mcp_bridge.load_mcp_server_class() is v2


def test_missing_package_says_not_installed(only_modules):
    only_modules({}, installed=())
    with pytest.raises(McpUnavailableError, match="not installed"):
        mcp_bridge.load_mcp_server_class()


def test_installed_but_unsupported_is_not_reported_as_missing(only_modules):
    """The old code printed 'not installed' when mcp 2.x was installed but lacked FastMCP."""
    only_modules({"mcp.server.fastmcp": {}}, installed=("mcp",))
    with pytest.raises(McpUnavailableError, match="no supported server class") as exc:
        mcp_bridge.load_mcp_server_class()
    assert "not installed" not in str(exc.value)
    assert "mcp.server.mcpserver.MCPServer" in str(exc.value)


def test_build_registers_every_service_tool(only_modules, tmp_path):
    only_modules({"mcp.server.mcpserver": {"MCPServer": _FakeServer}}, ("mcp",))
    service = OrchestratorService(tmp_path)
    server = mcp_bridge.build_mcp_server(service)
    assert server.name == mcp_bridge.SERVER_NAME
    assert set(server.tools) == set(service.tools())
    assert all(desc for _, desc in server.tools.values())
    assert mcp_bridge.build_fastmcp is mcp_bridge.build_mcp_server  # old name still works


def test_main_exits_2_with_the_specific_reason(only_modules, tmp_path, capsys):
    only_modules({}, installed=("mcp",))
    assert mcp_bridge.main(["--state-dir", str(tmp_path)]) == 2
    assert "no supported server class" in capsys.readouterr().err


def test_real_installed_mcp_lists_all_tools(tmp_path):
    pytest.importorskip("mcp")
    service = OrchestratorService(tmp_path)
    server = mcp_bridge.build_mcp_server(service)
    listed = {tool.name for tool in asyncio.run(server.list_tools())}
    assert listed == set(service.tools())
