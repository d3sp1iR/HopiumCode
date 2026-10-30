"""Цикл ответа: вопрос -> ретривер -> промпт -> LLM -> инструменты -> проверка ссылок.

Владелец: участник 1.

Агент не зависит от конкретных реализаций: модель, ретривер и инструменты передаются снаружи
(внедрение зависимостей). Поэтому его можно отлаживать на заглушках (см. stubs.py) и тестировать
без Ollama. Настоящие реализации подставляет build_default_agent().
"""
from __future__ import annotations

import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from .config import Config
from .models import Answer, Hit
from .prompts import ENRICHED_TEMPLATE, REPAIR_TEMPLATE, SYSTEM_PROMPT

log = logging.getLogger("hopiumcode")

# --------------------------------------------------------------------------- тип вопроса

_GRAPH_WORDS = ("кто вызывает", "что вызывает", "вызывается", "что сломается", "зависит", "зависимост",
                "наследу", "где используется", "кто использует", "импортирует", "callers", "what calls")
_HISTORY_WORDS = ("кто менял", "кто изменил", "кто последним", "когда изменил", "история", "коммит",
                  "почему так написано", "blame", "git log", "кто автор")
_SEMANTIC_WORDS = ("где ", "как работает", "как устроен", "объясни", "что делает", "для чего", "зачем")

_HINTS = {
    "graph": "Подсказка: это вопрос о связях в коде. Используй find_callers и граф-факты, а не догадки.",
    "history": "Подсказка: это вопрос об истории изменений. Используй git_log и git_blame.",
    "semantic": "",
    "other": "",
}


def classify(question: str) -> str:
    """Тип вопроса: graph | history | semantic | other. Порядок важен: связи и история сильнее смысловых слов."""
    q = question.lower().strip()
    if not q:
        return "other"
    if any(w in q for w in _GRAPH_WORDS):
        return "graph"
    if any(w in q for w in _HISTORY_WORDS):
        return "history"
    if any(w in q for w in _SEMANTIC_WORDS) or q.endswith("?"):
        return "semantic"
    return "other"


# --------------------------------------------------------------------------- проверка ссылок

_CITATION_RE = re.compile(r"\[([^\[\]:\n]+?):(\d+)(?:\s*[-–]\s*(\d+))?\]")
AllowedRef = tuple[str, int, int]  # (путь, начало, конец)


@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    good: list[str] = field(default_factory=list)
    bad: list[str] = field(default_factory=list)


def _norm_path(p: str) -> str:
    p = p.strip().replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return p


def parse_citations(text: str) -> list[tuple[str, int, int, str]]:
    """Все ссылки вида [путь:начало-конец] или [путь:строка]: (путь, начало, конец, исходный текст)."""
    found = []
    for m in _CITATION_RE.finditer(text):
        s = int(m.group(2))
        e = int(m.group(3)) if m.group(3) else s
        if e < s:
            s, e = e, s
        found.append((_norm_path(m.group(1)), s, e, m.group(0)))
    return found


def verify_citations(answer_text: str, allowed_refs: list[AllowedRef]) -> VerifyResult:
    """Ссылка допустима, если путь совпал и диапазон строк пересекается с допустимым диапазоном."""
    allowed = [(_norm_path(p), a, b) for p, a, b in allowed_refs]
    good: list[str] = []
    bad: list[str] = []
    for path, s, e, raw in parse_citations(answer_text):
        ok = any(path == p and s <= b and e >= a for p, a, b in allowed)
        (good if ok else bad).append(raw)
    return VerifyResult(ok=not bad, good=good, bad=bad)


# --------------------------------------------------------------------------- порты

@dataclass
class ToolCall:
    name: str
    args: dict


@dataclass
class ModelReply:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)


class ChatModel(Protocol):
    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> ModelReply: ...


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _get(obj, key, default=None):
    """Достаёт поле и из объекта, и из словаря (формат ответа Ollama зависит от версии пакета)."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class OllamaChat:
    """Адаптер к локальной модели через пакет ollama. Температура 0 для повторяемости оценки."""

    def __init__(self, model: str, host: str | None = None) -> None:
        import ollama  # импорт здесь, чтобы тесты работали без пакета

        self.model = model
        self._client = ollama.Client(host=host) if host else ollama.Client()

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> ModelReply:
        resp = self._client.chat(model=self.model, messages=messages, tools=tools or None,
                                 options={"temperature": 0})
        msg = _get(resp, "message")
        content = _THINK_RE.sub("", _get(msg, "content", "") or "").strip()
        calls = []
        for tc in _get(msg, "tool_calls", None) or []:
            fn = _get(tc, "function")
            args = _get(fn, "arguments", {}) or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            calls.append(ToolCall(_get(fn, "name"), dict(args)))
        return ModelReply(content, calls)


# --------------------------------------------------------------------------- агент

def _event(name: str, **kw) -> None:
    log.info(json.dumps({"event": name, **kw}, ensure_ascii=False, default=str))


def _hit_refs(hits: list[Hit]) -> list[AllowedRef]:
    return [(h.chunk.path, h.chunk.start_line, h.chunk.end_line) for h in hits]


def build_context(hits: list[Hit], max_chars: int = 6000) -> tuple[str, str, list[Hit]]:
    """Текст контекста для модели, граф-факты и список реально включённых фрагментов."""
    blocks: list[str] = []
    used: list[Hit] = []
    total = 0
    for h in hits:
        c = h.chunk
        block = f"{c.ref} ({c.kind} {c.name or ''}, найдено: {h.via})\n{c.text}".strip()
        if used and total + len(block) > max_chars:
            break
        blocks.append(block)
        used.append(h)
        total += len(block)
    context = "\n\n".join(blocks) if blocks else "(ничего не найдено)"
    facts = [f"- {h.chunk.name or h.chunk.id}: {h.via} {h.chunk.ref}" for h in used if h.via.startswith("graph:")]
    return context, "\n".join(facts) if facts else "(нет)", used


class Agent:
    def __init__(self, model: ChatModel, retrieve: Callable[[str, int, str], list[Hit]],
                 tools=None, config: Config | None = None, max_context_chars: int = 6000) -> None:
        self.model = model
        self.retrieve = retrieve
        self.tools = tools
        self.cfg = config or Config.from_env()
        self.max_context_chars = max_context_chars

    def _schemas(self) -> list[dict]:
        if self.tools is None:
            return []
        try:
            return self.tools.list_tools_for_ollama()
        except NotImplementedError:
            _event("tools_unavailable")
            return []

    def _call_tool(self, call: ToolCall) -> str:
        try:
            return str(self.tools.call_tool(call.name, call.args))
        except Exception as ex:  # noqa: BLE001 - любая ошибка инструмента возвращается модели текстом
            _event("tool_error", tool=call.name, error=repr(ex))
            return f"Ошибка инструмента {call.name}: {ex}"

    def answer(self, question: str, mode: str = "hybrid") -> Answer:
        t0 = time.perf_counter()
        qtype = classify(question)
        hits = self.retrieve(question, self.cfg.top_k, mode)
        context, facts, used_hits = build_context(hits, self.max_context_chars)
        allowed: list[AllowedRef] = _hit_refs(used_hits)
        _event("retrieved", qtype=qtype, mode=mode, hits=len(used_hits))

        user = ENRICHED_TEMPLATE.format(question=question, context_blocks=context, graph_facts=facts)
        if _HINTS[qtype]:
            user += "\n\n" + _HINTS[qtype]
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT.format(repo_name=self.cfg.repo_name)},
            {"role": "user", "content": user},
        ]
        schemas = self._schemas()
        used_tools: list[str] = []
        model_calls = 0
        final = ""

        for step in range(self.cfg.max_steps):
            reply = self.model.chat(messages, schemas or None)
            model_calls += 1
            if not reply.tool_calls:
                final = reply.content
                break
            messages.append({"role": "assistant", "content": reply.content,
                             "tool_calls": [{"function": {"name": c.name, "arguments": c.args}} for c in reply.tool_calls]})
            for call in reply.tool_calls:
                result = self._call_tool(call)
                used_tools.append(call.name)
                _event("tool_call", step=step, tool=call.name, args=call.args, result_chars=len(result))
                # ссылки, вернувшиеся от инструментов, тоже считаются источником
                allowed += [(p, s, e) for p, s, e, _ in parse_citations(result)]
                if call.name == "read_file" and "path" in call.args:
                    a, b = call.args.get("start_line") or 1, call.args.get("end_line") or 10**9
                    allowed.append((str(call.args["path"]), int(a), int(b)))
                messages.append({"role": "tool", "tool_name": call.name, "content": result})
        else:
            # лимит шагов исчерпан: просим окончательный ответ без инструментов
            messages.append({"role": "user", "content": "Лимит вызовов инструментов исчерпан. Дай ответ по тому, что уже найдено."})
            final = self.model.chat(messages, None).content
            model_calls += 1

        v = verify_citations(final, allowed)
        if not v.ok:  # одна попытка исправления
            _event("repair", bad=v.bad)
            messages += [{"role": "assistant", "content": final},
                         {"role": "user", "content": REPAIR_TEMPLATE.format(bad=", ".join(v.bad))}]
            final = self.model.chat(messages, None).content
            model_calls += 1
            v = verify_citations(final, allowed)

        text = final
        total = len(v.good) + len(v.bad)
        if v.bad:
            text += "\n\n⚠ Непроверенные ссылки (нет в найденном контексте): " + ", ".join(v.bad)
        if total:
            text += f"\n\nПодтверждено ссылок: {len(v.good)} из {total}."
        latency = time.perf_counter() - t0
        _event("answer", steps=model_calls, tools=used_tools, good=len(v.good), bad=len(v.bad), latency_s=round(latency, 2))
        return Answer(text=text, citations=v.good, bad_citations=v.bad, used_tools=used_tools,
                      steps=model_calls, latency_s=latency)


# --------------------------------------------------------------------------- мягкий откат на заглушки

_FALLBACKS: dict[str, str] = {}  # модуль -> текст предупреждения


def warn_fallback(module: str, ex: NotImplementedError, stub: str) -> None:
    """Один раз на модуль: предупреждение в stderr, что модуль ещё не готов и работает заглушка."""
    if module in _FALLBACKS:
        return
    owner = str(ex) or "владелец не указан"
    msg = f"⚠ {module} ещё не реализован ({owner}), используется заглушка {stub}"
    _FALLBACKS[module] = msg
    print(msg, file=sys.stderr)
    _event("fallback", module=module, owner=owner, stub=stub)


def fallback_warnings() -> list[str]:
    return list(_FALLBACKS.values())


def retrieve_with_fallback(real: Callable[[str, int, str], list[Hit]],
                           stub: Callable[[str, int, str], list[Hit]],
                           module: str = "retrieval.hybrid.retrieve",
                           stub_name: str = "stubs.stub_retrieve") -> Callable[[str, int, str], list[Hit]]:
    """Настоящий ретривер; после первого NotImplementedError навсегда переключается на заглушку."""
    use_stub = False

    def retrieve(question: str, k: int = 8, mode: str = "hybrid") -> list[Hit]:
        nonlocal use_stub
        if not use_stub:
            try:
                return real(question, k, mode)
            except NotImplementedError as ex:
                warn_fallback(module, ex, stub_name)
                use_stub = True
        return stub(question, k, mode)

    return retrieve


class _NoTools:
    """Инструменты недоступны (сервер не запустился): пустой список схем, вызов возвращает текст ошибки."""

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def start(self) -> None: ...

    def list_tools_for_ollama(self) -> list[dict]:
        return []

    def call_tool(self, name: str, args: dict) -> str:
        return f"Ошибка: инструменты недоступны ({self.reason})"

    def close(self) -> None: ...


class ToolsWithFallback:
    """Клиент инструментов (ToolClientLike) с откатом: пока настоящий не реализован, работает заглушка.

    Настоящий клиент запускается при первом обращении, поэтому Agent не обязан вызывать start().
    """

    def __init__(self, real, stub, module: str = "mcp_client.McpClient",
                 stub_name: str = "stubs.StubTools") -> None:
        self._real, self._stub = real, stub
        self._module, self._stub_name = module, stub_name
        self._active = None

    def _switch_to_stub(self, ex: NotImplementedError):
        warn_fallback(self._module, ex, self._stub_name)
        self._stub.start()
        self._active = self._stub
        return self._stub

    def _tools(self):
        if self._active is None:
            try:
                self._real.start()
                self._active = self._real
            except NotImplementedError as ex:
                self._switch_to_stub(ex)
            except Exception as ex:  # noqa: BLE001 - сервер не поднялся: отвечаем без инструментов, а не падаем
                msg = f"⚠ MCP-сервер не запустился ({ex}); агент отвечает без инструментов"
                print(msg, file=sys.stderr)
                _event("tools_start_failed", error=repr(ex))
                self._active = _NoTools(str(ex))
        return self._active

    def _call(self, method: str, *args):
        tools = self._tools()
        try:
            return getattr(tools, method)(*args)
        except NotImplementedError as ex:
            if tools is self._stub:
                raise
            return getattr(self._switch_to_stub(ex), method)(*args)

    def start(self) -> None:
        self._tools()

    def list_tools_for_ollama(self) -> list[dict]:
        return self._call("list_tools_for_ollama")

    def call_tool(self, name: str, args: dict) -> str:
        return self._call("call_tool", name, args)

    def close(self) -> None:
        if self._active is not None:
            try:
                self._active.close()
            except NotImplementedError:
                pass
            self._active = None


# --------------------------------------------------------------------------- сборка по умолчанию

def build_default_agent(config: Config | None = None) -> Agent:
    """Ollama + retrieve (участник 3) + MCP-клиент (участник 4); пока модуль не готов, работает его заглушка."""
    cfg = config or Config.from_env()
    from .mcp_client import McpClient
    from .retrieval.hybrid import retrieve
    from .stubs import StubTools, stub_retrieve

    return Agent(OllamaChat(cfg.llm_model, cfg.ollama_url), retrieve_with_fallback(retrieve, stub_retrieve),
                 ToolsWithFallback(McpClient(), StubTools()), cfg)


def build_fake_agent(config: Config | None = None) -> Agent:
    """Всё на заглушках и без Ollama: режим --fake и тесты."""
    from .stubs import ScriptedChat, StubTools, stub_retrieve

    return Agent(ScriptedChat(), stub_retrieve, StubTools(), config or Config.from_env())


def answer(question: str, mode: str = "hybrid") -> Answer:
    """Главная функция агента: используется консолью и скриптом оценки (участник 5)."""
    return build_default_agent().answer(question, mode)
