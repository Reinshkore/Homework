/**
 * 本地网页调用阿里云百炼 Managed Agents（AgentStudio）示例
 *
 * 架构：浏览器 -> 本地代理(Node) -> 百炼 Managed Agents API
 * 依赖：无（Node 18+ 内置 fetch / http）
 * 运行：node server.mjs   然后打开 http://localhost:3000
 *
 * 为什么需要本地代理：
 *   1) 浏览器直连百炼会触发跨域（CORS）拦截；
 *   2) API Key 放在前端会直接暴露，必须留在服务端。
 *
 * 官方文档：
 *   - API 总览与认证   https://help.aliyun.com/zh/model-studio/managed-agents-api-overview
 *   - 会话事件流（SSE） https://help.aliyun.com/zh/model-studio/managed-agents-event-stream
 *   - 订阅 Event 流     https://help.aliyun.com/zh/model-studio/event-sse-stream
 */
import http from 'node:http';
import { readFileSync, existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

/* ---------- 1. 读取 .env（零依赖迷你解析器） ---------- */
function loadEnv(file) {
  if (!existsSync(file)) return;
  for (const raw of readFileSync(file, 'utf8').split(/\r?\n/)) {
    const line = raw.trim();
    if (!line || line.startsWith('#')) continue;
    const eq = line.indexOf('=');
    if (eq === -1) continue;
    const key = line.slice(0, eq).trim();
    let val = line.slice(eq + 1).trim();
    if (
      (val.startsWith('"') && val.endsWith('"')) ||
      (val.startsWith("'") && val.endsWith("'"))
    ) {
      val = val.slice(1, -1);
    }
    if (!(key in process.env)) process.env[key] = val; // 真实环境变量优先于 .env
  }
}
loadEnv(path.join(__dirname, '.env'));

/* ---------- 2. 配置 ---------- */
const API_KEY = process.env.DASHSCOPE_API_KEY || '';
const WORKSPACE = process.env.BAILIAN_WORKSPACE_ID || '';
const REGION = process.env.BAILIAN_REGION || 'cn-beijing';
const SESSION = process.env.BAILIAN_SESSION_ID || '';
const PORT = Number(process.env.PORT || 3000);

// Managed Agents 基地址按「工作空间 + 地域」拼装
const BASE = `https://${WORKSPACE}.${REGION}.maas.aliyuncs.com/api/v1/agentstudio`;

const missing = [];
if (!API_KEY) missing.push('DASHSCOPE_API_KEY');
if (!WORKSPACE) missing.push('BAILIAN_WORKSPACE_ID');
if (!SESSION) missing.push('BAILIAN_SESSION_ID');

/* ---------- 3. 工具函数 ---------- */
function sendJson(res, code, obj) {
  res.writeHead(code, { 'Content-Type': 'application/json; charset=utf-8' });
  res.end(JSON.stringify(obj));
}

/**
 * 处理上游 SSE 的一个数据块（以空行分隔），返回 true 表示本轮对话结束
 */
function handleSseBlock(block, emit, streamedIds) {
  const dataLines = [];
  for (const line of block.split('\n')) {
    if (line.startsWith('data:')) dataLines.push(line.slice(5).trimStart());
    // 其余为 event: / id: / :注释（: connected、:keepalive），忽略
  }
  if (!dataLines.length) return false;

  let ev;
  try {
    ev = JSON.parse(dataLines.join('\n'));
  } catch {
    return false;
  }

  switch (ev.type) {
    case 'event_start':
      // 开启了 event_deltas 的事件，先记住其 id，用于与最终完整消息去重
      if (ev.event?.type === 'message' && ev.event.id) streamedIds.add(ev.event.id);
      return false;

    case 'event_delta': {
      const text = ev.delta?.content?.text;
      if (text) emit({ type: 'delta', text });
      return false;
    }

    case 'message': {
      if (ev.role !== 'assistant') return false;
      if (streamedIds.has(ev.id)) return false; // 已通过增量帧下发，跳过完整消息
      const text = (ev.content || [])
        .filter((b) => b.type === 'text')
        .map((b) => b.text)
        .join('');
      if (text) emit({ type: 'delta', text });
      return false;
    }

    case 'tool_call': {
      const name = ev.content?.[0]?.data?.name;
      if (name) emit({ type: 'info', text: `调用工具：${name}` });
      return false;
    }

    case 'session_status': {
      const d = ev.content?.[0]?.data || {};
      emit({ type: 'status', status: d.session_status, stop_reason: d.stop_reason });
      return d.session_status === 'idle' || d.session_status === 'terminated';
    }

    case 'error':
      emit({ type: 'error', message: ev.error?.message || '未知运行期错误' });
      return false;

    default:
      return false;
  }
}

/* ---------- 4. /api/chat：向浏览器转发 SSE ---------- */
async function handleChat(req, res) {
  if (missing.length) {
    return sendJson(res, 500, {
      message: `缺少配置：${missing.join('、')}，请在 .env 中填写后重启服务`,
    });
  }

  let body = '';
  for await (const chunk of req) body += chunk;
  let prompt = '';
  try {
    prompt = String(JSON.parse(body || '{}').prompt || '').trim();
  } catch {
    /* 忽略非法 JSON */
  }
  if (!prompt) return sendJson(res, 400, { message: 'prompt 不能为空' });

  // 对浏览器开启 SSE
  res.writeHead(200, {
    'Content-Type': 'text/event-stream; charset=utf-8',
    'Cache-Control': 'no-cache, no-transform',
    Connection: 'keep-alive',
  });
  const emit = (obj) => res.write(`data: ${JSON.stringify(obj)}\n\n`);

  const ac = new AbortController();
  req.on('close', () => ac.abort());

  try {
    // 关键顺序：先建立事件流长连接，再发送消息，否则可能错过早期事件
    const streamUrl =
      `${BASE}/sessions/${encodeURIComponent(SESSION)}/events/stream?` +
      new URLSearchParams([['event_deltas[]', 'message']]).toString();

    const upstream = await fetch(streamUrl, {
      headers: { Authorization: `Bearer ${API_KEY}`, Accept: 'text/event-stream' },
      signal: ac.signal,
    });

    if (!upstream.ok || !upstream.body) {
      const detail = await upstream.text().catch(() => '');
      emit({ type: 'error', message: `订阅事件流失败 HTTP ${upstream.status} ${detail.slice(0, 300)}` });
      return res.end();
    }

    // 再发送用户消息（同一会话即自动延续多轮上下文）
    const postRes = await fetch(`${BASE}/sessions/${encodeURIComponent(SESSION)}/events`, {
      method: 'POST',
      headers: { Authorization: `Bearer ${API_KEY}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({
        input: [
          { role: 'user', type: 'message', content: [{ type: 'text', text: prompt }] },
        ],
      }),
      signal: ac.signal,
    });

    if (!postRes.ok) {
      const detail = await postRes.text().catch(() => '');
      emit({ type: 'error', message: `发送消息失败 HTTP ${postRes.status} ${detail.slice(0, 300)}` });
      return res.end();
    }

    // 解析上游 SSE 并逐块转发
    const reader = upstream.body.getReader();
    const decoder = new TextDecoder();
    const streamedIds = new Set();
    let buffer = '';
    let finished = false;

    while (!finished) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let idx;
      while ((idx = buffer.indexOf('\n\n')) !== -1) {
        const block = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        if (handleSseBlock(block, emit, streamedIds)) {
          finished = true;
          break;
        }
      }
    }

    emit({ type: 'done' });
    res.end();
  } catch (err) {
    if (!ac.signal.aborted) {
      try {
        emit({ type: 'error', message: `本地代理异常：${String(err)}` });
      } catch {
        /* 连接可能已关闭 */
      }
    }
    res.end();
  }
}

/* ---------- 5. HTTP 服务 ---------- */
const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://localhost:${PORT}`);

  if (req.method === 'GET' && (url.pathname === '/' || url.pathname === '/index.html')) {
    try {
      const html = await readFile(path.join(__dirname, 'public', 'index.html'));
      res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
      return res.end(html);
    } catch {
      return sendJson(res, 500, { message: '未找到 public/index.html' });
    }
  }

  // 供前端展示当前配置状态
  if (req.method === 'GET' && url.pathname === '/api/meta') {
    return sendJson(res, 200, {
      sessionId: SESSION || null,
      hasKey: Boolean(API_KEY),
      hasWorkspace: Boolean(WORKSPACE),
      region: REGION,
    });
  }

  if (req.method === 'POST' && url.pathname === '/api/chat') {
    return handleChat(req, res);
  }

  sendJson(res, 404, { message: 'Not Found' });
});

server.listen(PORT, () => {
  console.log(`\n本地服务已启动：http://localhost:${PORT}`);
  if (missing.length) {
    console.log(`⚠ 缺少配置：${missing.join('、')}（请编辑 .env 后重启服务）`);
  } else {
    console.log(`会话：${SESSION}`);
    console.log(`端点：${BASE}`);
  }
});
