// ふぇれ子の中継（Cloudflare Worker）。
// スマホのアプリはこの中継だけを呼び、Anthropic の API キーはここ（Worker のシークレット）にだけ置く。
//
//   POST /v1/chat   {system, messages, model?}  -> Claude の SSE をそのまま流す
//   POST /v1/warm   {system, messages, model?}  -> キャッシュだけ先に作る（max_tokens 0、返事なし）
//   GET  /v1/usage                      -> 今日（日本時間）の使用額
//   どちらも Authorization: Bearer <RELAY_TOKEN> が必要
//
// SDK を使わず HTTP を直接呼ぶのは、無料プランの CPU 上限（1 回 10ms）に収めるため。
// 返事の流れは中身を読み直さずにそのまま渡し、使用量の計算に要る
// message_start と message_delta だけを読む。

// アプリが選べる頭（model）。考える量はそれぞれいちばん少ない設定にする（返事の出だしを速くするため）。
// Opus 5.5 は考えるのを止められないので effort low だけ。Sonnet 5.5 は between_tools、Haiku 5.5 は disabled。
// 定価（USD / 100 万トークン）：入力、出力、キャッシュ読み、5 分 / 1 時間キャッシュ書き。
const MODELS = {
  opus: { id: "claude-opus-5-5", price: { input: 4.0, output: 20.0, cacheRead: 0.2, write5m: 5.0, write1h: 8.0 } },
  sonnet: { id: "claude-sonnet-5-5", thinking: { type: "between_tools" }, price: { input: 2.0, output: 10.0, cacheRead: 0.1, write5m: 2.5, write1h: 4.0 } },
  haiku: { id: "claude-haiku-5-5", thinking: { type: "disabled" }, price: { input: 0.1, output: 0.5, cacheRead: 0.01, write5m: 0.125, write1h: 0.2 } },
};
const DEFAULT_MODEL = "opus";
const MAX_TOKENS = 2000;
// 1 時間キャッシュ：書き込みは 2 倍だが、しばらく空けて話しかけても最初の返事が遅くならない
const CACHE = { type: "ephemeral", ttl: "1h" };
const MAX_BODY_BYTES = 512 * 1024;
const MAX_SYSTEM_CHARS = 8000;

export default {
  async fetch(req, env, ctx) {
    const url = new URL(req.url);
    if (!(await authorized(req, env))) return json({ error: "unauthorized" }, 401);
    if (url.pathname === "/v1/usage" && req.method === "GET") return json(await readUsage(env));
    if (url.pathname === "/v1/chat" && req.method === "POST") return chat(req, env, ctx, false);
    if (url.pathname === "/v1/warm" && req.method === "POST") return chat(req, env, ctx, true);
    return json({ error: "not found" }, 404);
  },
};

async function authorized(req, env) {
  const expected = env.RELAY_TOKEN;
  if (!expected) return false; // シークレット未設定なら誰も通さない
  const got = (req.headers.get("authorization") || "").replace(/^Bearer\s+/i, "");
  const enc = new TextEncoder();
  const [a, b] = await Promise.all([
    crypto.subtle.digest("SHA-256", enc.encode(got)),
    crypto.subtle.digest("SHA-256", enc.encode(expected)),
  ]);
  return crypto.subtle.timingSafeEqual(a, b);
}

async function chat(req, env, ctx, warm) {
  if (!env.ANTHROPIC_API_KEY) return json({ error: "relay has no API key" }, 503);
  const raw = await req.text();
  if (raw.length > MAX_BODY_BYTES) return json({ error: "body too large" }, 413);
  let body;
  try {
    body = JSON.parse(raw);
  } catch {
    return json({ error: "invalid json" }, 400);
  }
  const { system, messages } = body;
  if (typeof system !== "string" || system.length > MAX_SYSTEM_CHARS) return json({ error: "bad system" }, 400);
  if (!Array.isArray(messages) || messages.length === 0 || messages[messages.length - 1].role !== "user") {
    return json({ error: "messages must end with a user turn" }, 400);
  }
  const key = body.model === undefined ? DEFAULT_MODEL : body.model;
  const m = MODELS[key];
  if (!m) return json({ error: "unknown model" }, 400);
  // Opus 以外は Opus の考えたブロック（署名つき）を読めないので、過去の返事から外して送る
  const sent = m.thinking
    ? messages.map((x) =>
        x.role === "assistant" && Array.isArray(x.content)
          ? { ...x, content: x.content.filter((b) => b.type !== "thinking" && b.type !== "redacted_thinking") }
          : x,
      )
    : messages;

  const usage = await readUsage(env);
  if (usage.usd >= usage.limit_usd) return json({ error: "daily budget reached", ...usage }, 429);

  // UPSTREAM はローカル試験で偽サーバーに向けるときだけ .dev.vars で設定する
  const upstream = await fetch(`${env.UPSTREAM || "https://api.anthropic.com"}/v1/messages`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "x-api-key": env.ANTHROPIC_API_KEY,
      "anthropic-version": "2023-06-01",
      "anthropic-beta": "server-side-fallback-2026-07-01",
    },
    // 考え方と effort は本番と温めで同じにする（違うとキャッシュが別物になる）
    body: JSON.stringify({
      model: m.id,
      max_tokens: warm ? 0 : MAX_TOKENS,
      ...(warm ? {} : { stream: true }),
      system: [{ type: "text", text: system, cache_control: CACHE }],
      messages: sent,
      ...(m.thinking ? { thinking: m.thinking } : {}),
      output_config: { effort: "low" },
      cache_control: CACHE,
      ...(warm ? {} : { fallbacks: "default" }),
    }),
  });
  if (!upstream.ok || !upstream.body) {
    // エラーの中身（API のエラー JSON）はそのまま返す。キーは含まれない。
    return new Response(await upstream.text(), {
      status: upstream.status,
      headers: { "content-type": upstream.headers.get("content-type") || "application/json" },
    });
  }

  if (warm) {
    const res = await upstream.json();
    await addUsage(env, cost(res.usage || {}, m, res.model), res.model || m.id, m.id);
    return json({ ok: true, model: res.model, usage: res.usage });
  }
  const [toClient, toMeter] = upstream.body.tee();
  ctx.waitUntil(meter(toMeter, env, m));
  return new Response(toClient, {
    headers: { "content-type": "text/event-stream", "cache-control": "no-store" },
  });
}

// message_start（入力とキャッシュ）と message_delta（出力）だけ読んで、今日の使用額に足す。
async function meter(stream, env, m) {
  const reader = stream.pipeThrough(new TextDecoderStream()).getReader();
  let buf = "";
  let model = m.id;
  const u = { input_tokens: 0, output_tokens: 0, cache_read_input_tokens: 0, cache_creation_input_tokens: 0 };
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += value;
    let cut;
    while ((cut = buf.indexOf("\n\n")) >= 0) {
      const ev = buf.slice(0, cut);
      buf = buf.slice(cut + 2);
      if (!ev.startsWith("event: message_start") && !ev.startsWith("event: message_delta")) continue;
      const line = ev.split("\n").find((l) => l.startsWith("data: "));
      if (!line) continue;
      const data = JSON.parse(line.slice(6));
      const usage = data.type === "message_start" ? data.message?.usage : data.usage;
      if (data.type === "message_start" && data.message?.model) model = data.message.model;
      if (!usage) continue;
      for (const k of Object.keys(u)) if (typeof usage[k] === "number") u[k] = usage[k]; // 累計値が来るので上書き
      if (usage.cache_creation) u.cache_creation = usage.cache_creation;
    }
  }
  await addUsage(env, cost(u, m, model), model, m.id);
}

// 使用量 → USD。キャッシュ書き込みは内訳（5 分 / 1 時間）があればそれで、なければ全部 1 時間として数える。
function cost(u, m, actual) {
  const p = (Object.values(MODELS).find((x) => x.id === actual) || m).price;   // 代わりのモデルで答えたときはその値段
  const n = (k) => (typeof u[k] === "number" ? u[k] : 0);
  const cc = u.cache_creation;
  const w5 = cc && typeof cc.ephemeral_5m_input_tokens === "number" ? cc.ephemeral_5m_input_tokens : 0;
  const w1 = cc && typeof cc.ephemeral_1h_input_tokens === "number" ? cc.ephemeral_1h_input_tokens : n("cache_creation_input_tokens") - w5;
  return (n("input_tokens") * p.input + n("output_tokens") * p.output + n("cache_read_input_tokens") * p.cacheRead + w5 * p.write5m + w1 * p.write1h) / 1e6;
}

function today() {
  return new Date(Date.now() + 9 * 3600 * 1000).toISOString().slice(0, 10); // 日本時間の日付
}

async function readUsage(env) {
  const limit = Number(env.DAILY_USD || "3");
  if (!env.USAGE) return { day: today(), usd: 0, requests: 0, limit_usd: limit, metering: "off" };
  const v = (await env.USAGE.get(today(), "json")) || { usd: 0, requests: 0 };
  return { day: today(), ...v, limit_usd: limit };
}

async function addUsage(env, usd, model, requested) {
  if (!env.USAGE) return;
  const day = today();
  const v = (await env.USAGE.get(day, "json")) || { usd: 0, requests: 0 };
  v.usd = Math.round((v.usd + usd) * 1e6) / 1e6;
  v.requests += 1;
  if (model !== requested) v.other_model = (v.other_model || 0) + 1;
  await env.USAGE.put(day, JSON.stringify(v), { expirationTtl: 60 * 60 * 24 * 40 });
}

function json(obj, status = 200) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" },
  });
}
