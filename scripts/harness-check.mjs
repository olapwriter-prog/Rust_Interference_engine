// Replicates the request/stream handling of stitaP's OpenAI provider (agent/src/providers/openai.ts)
// against rie-server, and runs a short agent loop with harness-style tools.
const BASE = process.env.RIE_BASE ?? "http://127.0.0.1:8091/v1";
const tools = [
  { name: "read_file", description: "Read a text file", parameters: { type: "object", properties: { path: { type: "string" } }, required: ["path"] } },
  { name: "terminal", description: "Run a shell command", parameters: { type: "object", properties: { command: { type: "string" }, timeout: { type: "integer" } }, required: ["command"] } },
  { name: "write_file", description: "Write a file", parameters: { type: "object", properties: { path: { type: "string" }, content: { type: "string" } }, required: ["path", "content"] } },
];
async function chat(messages, extra = {}) {
  const body = { model: "tiny-qwen", messages, temperature: 0.3, stream: true, max_tokens: 400,
    tools: tools.map((t) => ({ type: "function", function: t })), tool_choice: "auto", stream_options: { include_usage: true }, ...extra };
  const res = await fetch(BASE + "/chat/completions", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
  // SSE parsing as in sseEvents(): data lines, blank line terminates an event
  const text = await res.text();
  let content = "", finish = "", usage, timings, id; const calls = [];
  for (const block of text.split("\n\n")) {
    const data = block.split("\n").filter((l) => l.startsWith("data:")).map((l) => l.slice(5).trimStart()).join("\n");
    if (!data || data === "[DONE]") continue;
    const j = JSON.parse(data); id = j.id;
    if (j.usage) usage = j.usage; if (j.timings) timings = j.timings;
    const d = j.choices?.[0]?.delta ?? {};
    if (d.content) content += d.content;
    for (const tc of d.tool_calls ?? []) {   // merge by index like readStream()
      const c = (calls[tc.index ?? 0] ??= { id: "", name: "", arguments: "" });
      if (tc.id) c.id = tc.id; if (tc.function?.name) c.name += tc.function.name; if (tc.function?.arguments) c.arguments += tc.function.arguments;
    }
    if (j.choices?.[0]?.finish_reason) finish = j.choices[0].finish_reason;
  }
  return { id, content, calls, finish, usage, timings };
}
const fail = (m) => { console.error("FAIL:", m); process.exit(1); };
const messages = [{ role: "system", content: "You are a coding agent." }, { role: "user", content: "Show me the README." }];
// turn 1: tool_choice auto (what the harness sends) -> any well-formed reply
let r = await chat(messages);
if (!r.usage || !r.timings || typeof r.timings.predicted_per_second !== "number") fail("missing usage/timings");
console.log(`turn 1 (auto): finish=${r.finish} calls=${r.calls.length} gen ${r.timings.predicted_per_second.toFixed(0)} tok/s, prompt ${r.usage.prompt_tokens} tok`);
// turn 2: force a call (a random-weight model rarely chooses one by itself)
r = await chat(messages, { tool_choice: "required" });
if (r.finish !== "tool_calls" || r.calls.length !== 1) fail("expected a tool call: " + JSON.stringify(r));
const call = r.calls[0]; const args = JSON.parse(call.arguments);
const t = tools.find((x) => x.name === call.name); if (!t) fail("unknown tool " + call.name);
for (const k of t.parameters.required) if (!(k in args)) fail(`missing ${k}`);
console.log(`turn 2 (required): ${call.name}(${call.arguments.slice(0, 80)}) id=${call.id}`);
// turn 3: the harness executes the tool and sends the result back (an error here)
messages.push({ role: "assistant", content: r.content || null, tool_calls: [{ id: call.id, type: "function", function: { name: call.name, arguments: call.arguments } }] });
messages.push({ role: "tool", tool_call_id: call.id, name: call.name, content: "Error: no such file or directory" });
const before = await (await fetch(BASE + "/rl")).json();
r = await chat(messages);
const after = await (await fetch(BASE + "/rl")).json();
console.log(`turn 3: finish=${r.finish}, prompt reused ${r.timings.cache_n} cached tokens; RL updates ${before.updates} -> ${after.updates}, feedback events ${before.feedback_events} -> ${after.feedback_events}`);
if (after.feedback_events <= before.feedback_events) fail("auto-reward did not fire on the tool error");
// explicit feedback + correction on the last completion
const fb = await (await fetch(BASE + "/feedback", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ id: r.id, reward: 1.0 }) })).json();
const cr = await (await fetch(BASE + "/corrections", { method: "POST", headers: { "content-type": "application/json" },
  body: JSON.stringify({ id: r.id, tool_call: { name: "terminal", arguments: { command: "ls" } } }) })).json();
console.log(`feedback: ${JSON.stringify(fb)}; correction: ${JSON.stringify(cr)}`);
if (!fb.ok || !cr.ok) fail("feedback/correction failed");
console.log("OK: harness-style agent loop works against rie-server");
