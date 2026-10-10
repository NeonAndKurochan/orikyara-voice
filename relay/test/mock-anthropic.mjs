// ローカル試験用の偽 Anthropic API。本物の API には一切つながない。
import http from "node:http";

const port = Number(process.argv[2] || 8799);
http
  .createServer((req, res) => {
    let body = "";
    req.on("data", (c) => (body += c));
    req.on("end", () => {
      const ok = req.headers["x-api-key"] === "test-key" && req.headers["anthropic-beta"] === "server-side-fallback-2026-07-01";
      const parsed = JSON.parse(body);
      console.error(
        `mock: model=${parsed.model} max_tokens=${parsed.max_tokens} stream=${parsed.stream} thinking=${JSON.stringify(parsed.thinking)} ` +
          `effort=${parsed.output_config?.effort} cache=${JSON.stringify(parsed.cache_control)} system_cache=${JSON.stringify(parsed.system?.[0]?.cache_control)} ` +
          `fallbacks=${parsed.fallbacks} msgs=${parsed.messages.length} key_ok=${ok}`,
      );
      for (const m of parsed.messages) if (m.role === "assistant") console.error(`mock:   echoed assistant content ${JSON.stringify(m.content)}`);
      if (!ok) {
        res.writeHead(401, { "content-type": "application/json" });
        return res.end('{"type":"error","error":{"type":"authentication_error","message":"bad key"}}');
      }
      if (parsed.max_tokens === 0) {
        // pre-warm: no output, just the cache write
        res.writeHead(200, { "content-type": "application/json" });
        return res.end(JSON.stringify({ id: "msg_warm", model: parsed.model, content: [], stop_reason: "max_tokens",
          usage: { input_tokens: 4, output_tokens: 0, cache_read_input_tokens: 0, cache_creation_input_tokens: 3000,
            cache_creation: { ephemeral_5m_input_tokens: 0, ephemeral_1h_input_tokens: 3000 } } }));
      }
      res.writeHead(200, { "content-type": "text/event-stream" });
      const ev = (type, data) => `event: ${type}\ndata: ${JSON.stringify({ type, ...data })}\n\n`;
      const parts = [
        ev("message_start", { message: { id: "msg_mock", model: parsed.model, usage: { input_tokens: 1000, output_tokens: 1, cache_read_input_tokens: 2000, cache_creation_input_tokens: 0 } } }),
        ev("content_block_start", { index: 0, content_block: { type: "thinking", thinking: "", signature: "" } }),
        ev("content_block_delta", { index: 0, delta: { type: "signature_delta", signature: "sig123" } }),
        ev("content_block_stop", { index: 0 }),
        ev("content_block_start", { index: 1, content_block: { type: "text", text: "" } }),
        ev("content_block_delta", { index: 1, delta: { type: "text_delta", text: "やっほー！" } }),
        ev("content_block_delta", { index: 1, delta: { type: "text_delta", text: "元気だよ😊" } }),
        ev("content_block_stop", { index: 1 }),
        ev("message_delta", { delta: { stop_reason: "end_turn" }, usage: { output_tokens: 50 } }),
        ev("message_stop", {}),
      ];
      let i = 0;
      const tick = () => {
        if (i < parts.length) {
          res.write(parts[i++]);
          setTimeout(tick, 30);
        } else res.end();
      };
      tick();
    });
  })
  .listen(port, () => console.error(`mock listening ${port}`));
