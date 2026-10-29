"""MCP-сервер: запуск по stdio и инструменты как обычные функции. Без Ollama и клона Flask."""
import asyncio
import os
import sys

from hopiumcode.mcp_server import server


async def _list_tools_over_stdio(root):
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    params = StdioServerParameters(command=sys.executable, args=["-m", "hopiumcode.mcp_server.server"],
                                   env={"REPO_ROOT": str(root), "PYTHONPATH": os.pathsep.join(sys.path)})
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        return [t.name for t in (await session.list_tools()).tools]


def test_server_starts_over_stdio(tmp_path):
    names = asyncio.run(asyncio.wait_for(_list_tools_over_stdio(tmp_path), 60))
    assert isinstance(names, list)


def test_root_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("REPO_ROOT", str(tmp_path))
    assert server.get_root() == tmp_path.resolve()


def test_root_from_config(tmp_path, monkeypatch):
    monkeypatch.delenv("REPO_ROOT", raising=False)
    monkeypatch.setenv("CG_REPO_ROOT", str(tmp_path / "src"))
    assert server.get_root() == (tmp_path / "src").resolve()
