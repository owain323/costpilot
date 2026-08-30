"""P1-D1: discover_ai_calls: statically scan a codebase for LLM API call sites.

Design principles (high confidence over recall):
- High-confidence first: prefer missing ambiguous patterns over producing misleading results
  (the review's "high-confidence detection" requirement).
- Structured output: extract only certain facts (file/line/framework/model/confidence);
  never feed raw source to the agent.
- Explicit support scope: direct SDK invocations + known wrappers + Strands Agent construction;
  no dynamic-reflection chasing.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from .models import AICallSite, Framework


@dataclass
class ScanConfig:
    """Scan configuration: explicit support scope (non-goal: full coverage)."""

    include_suffixes: tuple[str, ...] = (".py",)
    skip_dirs: tuple[str, ...] = (
        ".git",
        "node_modules",
        "venv",
        ".venv",
        "dist",
        "build",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
    )
    max_call_expr_len: int = 120


# Framework matching rules: match against the AST call chain.
# Each rule: (framework, function name, required attrs, confidence)
_RULES = (
    # OpenAI Chat Completions: client.chat.completions.create(...)
    (
        Framework.OPENAI,
        "create",
        ("chat", "completions"),
        0.96,
    ),
    # OpenAI Responses API: client.responses.create(...)
    (
        Framework.OPENAI_RESPONSES,
        "create",
        ("responses",),
        0.95,
    ),
    # OpenAI Beta Chat Completions: client.beta.chat.completions.parse(...)
    (
        Framework.OPENAI,
        "parse",
        ("chat", "completions"),
        0.94,
    ),
    # Anthropic Messages: client.messages.create(...)
    (
        Framework.ANTHROPIC,
        "create",
        ("messages",),
        0.96,
    ),
)

# LangChain model constructor class names -> framework
_LANGCHAIN_MODEL_CLASSES = {
    "ChatOpenAI": (Framework.LANGCHAIN, 0.90),
    "ChatAnthropic": (Framework.LANGCHAIN, 0.90),
    "ChatGroq": (Framework.LANGCHAIN, 0.88),
    "ChatGoogleGenerativeAI": (Framework.LANGCHAIN, 0.88),
    "AzureChatOpenAI": (Framework.LANGCHAIN, 0.90),
}


def _func_chain(call: ast.Call) -> tuple[str, ...]:
    """Flatten a call expression into a name chain:
    client.chat.completions.create -> (client, chat, completions, create)"""
    parts: list[str] = []
    node: ast.AST = call.func
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return tuple(reversed(parts))


def _match_sdk_rule(chain: tuple[str, ...]) -> tuple[Framework, float] | None:
    """Match SDK rules by chain features (attr segment + trailing function name)."""
    if not chain:
        return None
    func_name = chain[-1]
    attrs = chain[:-1]  # attribute chain without the function name
    for framework, fname, required_attrs, conf in _RULES:
        if func_name != fname:
            continue
        # The attribute chain must contain the required features (e.g. chat/completions)
        if all(req in attrs for req in required_attrs):
            return framework, conf
    return None


def _match_langchain_rule(call: ast.Call) -> tuple[Framework, float] | None:
    if not isinstance(call.func, ast.Name):
        return None
    if call.func.id in _LANGCHAIN_MODEL_CLASSES:
        return _LANGCHAIN_MODEL_CLASSES[call.func.id]
    return None


def _collect_module_constants(tree: ast.Module) -> dict[str, str]:
    """Module-level string constants: MODEL = 'gpt-4o' style."""
    consts: dict[str, str] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            consts[node.targets[0].id] = node.value.value
    return consts


def _collect_param_defaults(func: ast.FunctionDef) -> dict[str, str]:
    """Function parameter string defaults: def foo(model='gpt-4o').

    Python default values align to the *rightmost* params, so match by offset
    (plain zip would pair the first arg with the first default: wrong).
    """
    defaults: dict[str, str] = {}
    offset = len(func.args.args) - len(func.args.defaults)
    for i, arg in enumerate(func.args.args):
        if i >= offset:
            default = func.args.defaults[i - offset]
            if isinstance(default, ast.Constant) and isinstance(default.value, str):
                defaults[arg.arg] = default.value
    return defaults


def _env_default(call: ast.Call) -> str | None:
    """os.getenv('MODEL', 'gpt-4o') / os.environ.get('MODEL', 'gpt-4o') default value."""
    chain = _func_chain(call)
    if (
        len(chain) == 2
        and chain[1] in ("getenv", "get")
        and len(call.args) >= 2
        and isinstance(call.args[1], ast.Constant)
        and isinstance(call.args[1].value, str)
    ):
        return call.args[1].value
    return None


# Model-source penalties (more indirect = lower confidence, never below 0.70)
_MODEL_SOURCE_PENALTY = {
    "literal": 0.00,
    "constant": 0.03,
    "param_default": 0.06,
    "env_default": 0.08,
    "unknown": 0.10,
}


def _resolve_model(
    call: ast.Call, module_consts: dict[str, str], func: ast.FunctionDef | None
) -> tuple[str | None, str]:
    """Resolve the model argument with explicit provenance.

    Returns (model, source) where source is one of:
      literal / constant / param_default / env_default / unknown
    Resolution is conservative: unresolvable -> (None, 'unknown'), never guessed.
    """
    for kw in call.keywords:
        if kw.arg != "model":
            continue
        value = kw.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value.value, "literal"
        if isinstance(value, ast.Name):
            if value.id in module_consts:
                return module_consts[value.id], "constant"
            if func is not None:
                param_defaults = _collect_param_defaults(func)
                if value.id in param_defaults:
                    return param_defaults[value.id], "param_default"
            return None, "unknown"
        if isinstance(value, ast.Call):
            env_model = _env_default(value)
            if env_model is not None:
                return env_model, "env_default"
        return None, "unknown"
    return None, "unknown"


def _extract_params(call: ast.Call) -> dict:
    """Extract key numeric params (max_tokens/temperature etc., constants only)."""
    params: dict = {}
    for kw in call.keywords:
        if (
            kw.arg in ("max_tokens", "max_output_tokens", "temperature", "top_p")
            and isinstance(kw.value, ast.Constant)
            and isinstance(kw.value.value, (int, float))
        ):
            params[kw.arg] = kw.value.value
    return params


def _estimate_input_chars(call: ast.Call) -> int:
    """Estimate the char length of messages/prompt args: count string lengths only,
    never retain content (prompt-injection defense)."""
    total = 0
    for kw in call.keywords:
        if kw.arg not in ("messages", "prompt", "content", "system"):
            continue
        for node in ast.walk(kw.value):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                total += len(node.value)
    return total


def _classify(call: ast.Call, chain: tuple[str, ...]) -> tuple[Framework, float] | None:
    sdk = _match_sdk_rule(chain)
    if sdk:
        return sdk
    return _match_langchain_rule(call)


def _collect_model_instance_vars(tree: ast.AST) -> set[str]:
    """Collect known LangChain model instance variables (llm = ChatOpenAI(...) form).

    Simplified scope handling (module + function level); name-collision probability
    is low and paired with the 0.85 confidence. This is the first step of D2
    structured facts (variable tracking).
    """
    model_vars: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and _match_langchain_rule(node.value)
        ):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    model_vars.add(t.id)
    return model_vars


def _collect_completions_vars(tree: ast.AST) -> tuple[set[str], set[str]]:
    """Collect variables assigned to a `*.chat.completions` object: the
    split-invocation pattern found in real repos:

        self.client = openai.OpenAI(**params).chat.completions
        ...
        response = self.client.create(**params)

    Returns (plain names, attribute tails) so a later create/parse call on the
    same variable can be recognized. Confidence is lower (0.88) than direct
    invocation because the object source is not verified as an OpenAI client.
    """
    names: set[str] = set()
    attr_tails: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        val = node.value
        if isinstance(val, ast.Attribute) and val.attr == "completions":
            base = val.value
            if isinstance(base, ast.Attribute) and base.attr == "chat":
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        names.add(t.id)
                    elif isinstance(t, ast.Attribute):
                        attr_tails.add(t.attr)
    return names, attr_tails


def _match_completions_var(
    call: ast.Call, chain: tuple[str, ...], names: set[str], attr_tails: set[str]
) -> tuple[Framework, float] | None:
    """Match `known_chat_completions_var.create(...)` / `.parse(...)`."""
    if len(chain) not in (2, 3):
        return None
    if chain[-1] not in ("create", "parse"):
        return None
    if chain[-2] in names or chain[-2] in attr_tails:
        return Framework.OPENAI, 0.88
    return None


# Call methods on LangChain model instances (invoke/predict family)
_LANGCHAIN_INVOKE_METHODS = ("invoke", "ainvoke", "predict", "apredict")


def _match_langchain_invoke(
    call: ast.Call, chain: tuple[str, ...], model_vars: set[str]
) -> tuple[Framework, float] | None:
    """Match known_model_var.invoke(...): LangChain calls needing variable tracking."""
    if len(chain) == 2 and chain[1] in _LANGCHAIN_INVOKE_METHODS and chain[0] in model_vars:
        return Framework.LANGCHAIN, 0.85
    return None


def _extract_call_expr(call: ast.Call, limit: int) -> str:
    """Extract the raw call expression (truncated) for reporting/diagnostics;
    not used for agent decisions."""
    try:
        return ast.unparse(call)[:limit]
    except Exception:
        return "<unparse-failed>"


def _build_parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    """Map each node to its parent AST node (for scope lookups)."""
    return {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}


def _nearest_function(node: ast.AST, parent: dict[ast.AST, ast.AST]) -> ast.FunctionDef | None:
    """Walk up from a node to the nearest enclosing function definition."""
    cur = parent.get(node)
    while cur is not None:
        if isinstance(cur, ast.FunctionDef):
            return cur
        cur = parent.get(cur)
    return None


def discover_ai_calls(repo_path: str | Path, config: ScanConfig | None = None) -> list[AICallSite]:
    """Scan a repository and return all detected LLM call sites (sorted by file/line).

    Returns structured facts only; file contents never enter the agent context
    (first step of the prompt-injection defense).
    """
    config = config or ScanConfig()
    root = Path(repo_path)
    sites: list[AICallSite] = []

    # Accept either a directory or a single file as input
    if root.is_file():
        targets: list[Path] = [root]
    else:
        targets = [p for p in root.rglob("*") if p.is_file()]

    for py_file in targets:
        # Filter by suffix and skip dirs
        if py_file.suffix not in config.include_suffixes:
            continue
        if any(part in config.skip_dirs for part in py_file.parts):
            continue

        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8", errors="replace"))
        except (SyntaxError, UnicodeDecodeError, OSError):
            continue  # not Python or corrupted: skip (explicit support scope)

        model_vars = _collect_model_instance_vars(tree)
        completions_names, completions_attr_tails = _collect_completions_vars(tree)
        module_consts = _collect_module_constants(tree)

        # Parent map to locate the enclosing function of each call site
        parent = _build_parent_map(tree)

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            chain = _func_chain(node)
            matched = _classify(node, chain)
            if matched is None:
                matched = _match_langchain_invoke(node, chain, model_vars)
            if matched is None:
                matched = _match_completions_var(
                    node, chain, completions_names, completions_attr_tails
                )
            if matched is None:
                continue
            framework, confidence = matched
            model, model_source = _resolve_model(
                node, module_consts, _nearest_function(node, parent)
            )
            # Report the line where the model kwarg actually lives, not the
            # enclosing Call node — otherwise patch/UI/refinement tools anchor
            # to the wrong line, and a regex-based row refinement can drift a
            # site to an unrelated `model="..."` (we previously saw
            # `model="gpt-4o"` match `model="gpt-4o-mini"` by prefix).
            model_line = node.lineno
            for kw in node.keywords:
                if kw.arg == "model":
                    model_line = kw.value.lineno
                    break
            source_penalty = _MODEL_SOURCE_PENALTY.get(model_source, 0.10)
            confidence = max(0.70, round(confidence - source_penalty, 2))
            sites.append(
                AICallSite(
                    file=(
                        py_file.name if root.is_file() else str(py_file.relative_to(root))
                    ).replace("\\", "/"),
                    line=model_line,
                    framework=framework,
                    model=model,
                    confidence=confidence,
                    call_expr=_extract_call_expr(node, config.max_call_expr_len),
                    params=_extract_params(node),
                    estimated_input_chars=_estimate_input_chars(node),
                    model_source=model_source,
                )
            )

    sites.sort(key=lambda s: s.sort_key)
    return sites
