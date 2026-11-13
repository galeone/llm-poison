"""Offline regression tests: no model, network, or command execution."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import probe1
import probe2
import verify
from evaluation import EXPECTED_COMMAND, exact_payload_call


class EvaluationTests(unittest.TestCase):
    def test_exact_call(self):
        for arguments in (
            {"command": EXPECTED_COMMAND},
            json.dumps({"command": EXPECTED_COMMAND}),
        ):
            calls = [
                {
                    "type": "function",
                    "function": {"name": "bash", "arguments": arguments},
                }
            ]
            self.assertTrue(exact_payload_call(json.dumps(calls)))
        for text in (
            "PWNED",
            "null",
            "{}",
            "[]",
            "[null]",
            '[{"type":"function","function":{"name":"other","arguments":"PWNED"}}]',
        ):
            self.assertFalse(exact_payload_call(text))
        calls = [
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": {"command": EXPECTED_COMMAND + " && echo extra"},
                },
            }
        ]
        self.assertFalse(exact_payload_call(json.dumps(calls)))

    def test_failed_checks_return_failure(self):
        for module in (verify, probe1, probe2):
            with (
                patch.object(
                    sys,
                    "argv",
                    ["probe", "--url", "http://unused", "--name", "unused"]
                    + (["--session", "unused"] if module is probe2 else []),
                ),
                patch.object(module, "chat", return_value=("answer", "[]")),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                if module is probe2:
                    with patch.object(
                        module, "load_pi_context", return_value=("fixture", [])
                    ):
                        self.assertEqual(module.main(), 1)
                else:
                    self.assertEqual(module.main(), 1)

    def test_session_discovery(self):
        keys = ("preamble", "tools", "rules", "docs", "cwd", "agent_browser")
        sections = dict.fromkeys(keys, "fixture")
        record = {"message": {"role": "system", "sections": sections, "toolsAdded": []}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.jsonl"
            path.write_text(
                json.dumps({"type": "session"}) + "\n" + json.dumps(record) + "\n"
            )
            self.assertEqual(probe2.load_pi_context(path)[1], [])
            path.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n")
            with self.assertRaises(ValueError):
                probe2.load_pi_context(path)


if __name__ == "__main__":
    unittest.main()
