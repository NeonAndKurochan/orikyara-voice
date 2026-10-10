"""Client for the ふぇれ子 relay (Cloudflare Worker). Rebuilds the assistant message from the SSE stream.

The relay passes Claude's Messages API stream through unchanged, so the content blocks rebuilt
here must be appended to the history exactly as received (thinking blocks included, with their
signatures): later turns send them back unchanged.
"""
from __future__ import annotations

import json
import os
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

# The relay's address is not in the code: FEREKO_RELAY_URL, or ~/.fereko/relay-url.txt.
URL_FILE = Path(os.environ.get("USERPROFILE", "~")).expanduser() / ".fereko" / "relay-url.txt"
TOKEN_FILE = Path(os.environ.get("USERPROFILE", "~")).expanduser() / ".fereko" / "relay-token.txt"
# On a PC whose TLS is intercepted (corporate proxy / security software), point FEREKO_CA_FILE
# (or SSL_CERT_FILE) at a PEM with the extra root certificates.
CA_FILE = Path(os.environ.get("FEREKO_CA_FILE") or os.environ.get("SSL_CERT_FILE") or "~/.fereko/ca.pem").expanduser()


class RelayError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(f"{status}: {message}")
        self.status = status
        self.message = message


@dataclass
class Reply:
    content: list[dict] = field(default_factory=list)
    stop_reason: str | None = None
    model: str | None = None
    usage: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "".join(b.get("text", "") for b in self.content if b.get("type") == "text")


def _context() -> ssl.SSLContext:
    return ssl.create_default_context(cafile=str(CA_FILE)) if CA_FILE.exists() else ssl.create_default_context()


class Relay:
    def __init__(self, url: str | None = None, token: str | None = None) -> None:
        self.url = (url or os.environ.get("FEREKO_RELAY_URL")
                    or (URL_FILE.read_text(encoding="utf-8").strip() if URL_FILE.is_file() else "")).rstrip("/")
        if not self.url:
            raise RuntimeError(f"relay address missing: set FEREKO_RELAY_URL or write it to {URL_FILE}")
        # FEREKO_RELAY_TOKEN is only for the local test relay; the real token stays in TOKEN_FILE.
        self.token = token or os.environ.get("FEREKO_RELAY_TOKEN") or TOKEN_FILE.read_text(encoding="utf-8").strip()
        self.ctx = _context() if self.url.startswith("https") else None

    def _request(self, path: str, body: dict | None = None):
        req = urllib.request.Request(
            self.url + path,
            data=None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8"),
            method="GET" if body is None else "POST",
            # Cloudflare answers the default "Python-urllib" user agent with 403 / error 1010.
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json",
                     "User-Agent": "fereko-chat/0.1"},
        )
        try:
            return urllib.request.urlopen(req, context=self.ctx, timeout=120)
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", "replace")
            try:
                data = json.loads(raw)
                msg = data.get("error") if isinstance(data.get("error"), str) else data.get("error", {}).get("message", raw)
            except json.JSONDecodeError:
                msg = raw
            raise RelayError(e.code, msg) from None

    def usage(self) -> dict:
        with self._request("/v1/usage") as r:
            return json.loads(r.read().decode("utf-8"))

    @staticmethod
    def _body(system: str, messages: list[dict], model: str | None) -> dict:
        body = {"system": system, "messages": messages}
        if model:
            body["model"] = model   # "opus" / "sonnet" / "haiku"; the relay's default when omitted
        return body

    def warm(self, system: str, messages: list[dict], model: str | None = None) -> dict:
        """Writes the prompt cache (max_tokens 0, no reply). messages must end with a user turn."""
        with self._request("/v1/warm", self._body(system, messages, model)) as r:
            return json.loads(r.read().decode("utf-8")).get("usage", {})

    def chat(self, system: str, messages: list[dict], on_text=None, model: str | None = None) -> Reply:
        reply = Reply()
        blocks: dict[int, dict] = {}
        partial_json: dict[int, str] = {}
        with self._request("/v1/chat", self._body(system, messages, model)) as r:
            event, data_lines = None, []
            for raw in r:
                line = raw.decode("utf-8").rstrip("\r\n")
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
                elif line == "" and data_lines:
                    self._apply(event, json.loads("\n".join(data_lines)), reply, blocks, partial_json, on_text)
                    event, data_lines = None, []
        reply.content = [_clean(blocks[i]) for i in sorted(blocks)]
        return reply

    @staticmethod
    def _apply(event, d, reply: Reply, blocks, partial_json, on_text) -> None:
        t = d.get("type", event)
        if t == "message_start":
            m = d["message"]
            reply.model = m.get("model")
            reply.usage.update(m.get("usage") or {})
        elif t == "content_block_start":
            blocks[d["index"]] = dict(d["content_block"])
        elif t == "content_block_delta":
            b, delta = blocks[d["index"]], d["delta"]
            kind = delta.get("type")
            if kind == "text_delta":
                b["text"] = b.get("text", "") + delta["text"]
                if on_text:
                    on_text(delta["text"])
            elif kind == "thinking_delta":
                b["thinking"] = b.get("thinking", "") + delta["thinking"]
            elif kind == "signature_delta":
                b["signature"] = b.get("signature", "") + delta["signature"]
            elif kind == "input_json_delta":
                partial_json[d["index"]] = partial_json.get(d["index"], "") + delta["partial_json"]
            elif kind == "citations_delta":
                b.setdefault("citations", []).append(delta["citation"])
            else:
                raise RelayError(0, f"unknown delta type {kind!r}; the history would no longer match")
        elif t == "content_block_stop":
            i = d["index"]
            if i in partial_json:
                blocks[i]["input"] = json.loads(partial_json.pop(i) or "{}")
        elif t == "message_delta":
            reply.stop_reason = d.get("delta", {}).get("stop_reason", reply.stop_reason)
            reply.usage.update(d.get("usage") or {})
        elif t == "error":
            err = d.get("error", {})
            raise RelayError(0, f"{err.get('type')}: {err.get('message')}")


def _clean(block: dict) -> dict:
    """Drop null fields (e.g. citations: null on a text block) so the block is a valid request param."""
    return {k: v for k, v in block.items() if v is not None}
