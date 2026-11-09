# Using RIE with the stitaP tool harness (or any tool-calling agent)

`rie-server` lets small on-device models (0.5B–8B) drive a tool harness reliably, and
lets them learn from their mistakes while they run.

## What it does differently from a plain model server

| Problem with small models in an agent loop | What `rie-server` does |
|---|---|
| Broken JSON, made-up tool names, misspelled argument names | **Grammar-constrained tool calls.** Once a call starts (`<tool_call>` for Qwen/Hermes-style models, a leading `{` for Llama 3), only tokens that keep it a valid call can be sampled: a declared tool name, declared argument names, the declared JSON types and enum values, required arguments present. Every call parses. |
| The model never decides to call a tool | `tool_choice: "required"` or a named function prefills the call and constrains it. |
| The same mistake again and again | **Online reinforcement learning.** A mixture of small correction experts on the output layer learns from rewards. The base model is never changed, every correction is bounded, and the experts can be reset or saved to a file. |
| Nobody gives explicit feedback | `--auto-reward`: when the harness sends a tool result back, an error (`Error: …`, `{"error": …}`, tracebacks, "no such file", …) gives the call that caused it a negative reward, and success gives a small positive one. |
| Re-sending the whole conversation every turn is slow on a phone | Prompt-prefix reuse: only the new tokens are evaluated (`timings.cache_n` reports how many were reused). |

## Start the server

```bash
cargo build --release -p rie-server
./target/release/rie-server -m qwen2.5-3b-instruct-q4_0.gguf -c 16384 --port 8080 \
    --auto-reward --adapter ~/.stitap/rie-adapter.riea
```

`--adapter FILE` keeps what was learned across restarts. Use `--rl` for in-memory
learning only, or leave both out to serve without learning.

## Option A: point the harness at it

`~/.stitap/config.yaml`:

```yaml
model:
  provider: openai
  base_url: http://127.0.0.1:8080/v1
  name: qwen2.5-3b
  tool_mode: native        # rie-server does native tool calling
  context_window: 16384
agent:
  tool_profile: slm        # ~10 tools is the sweet spot for ≤4B models
```

## Option B: let the harness launch it (model router)

The harness's router starts `llama-server` with llama.cpp flags (`-m -c -ngl --fit -fa
-ctk -ctv -t --mlock --jinja --metrics --host --port --alias`). `rie-server` accepts all
of them (`-ctk q8_0` turns on the 8-bit KV cache), so you can point the router at it:

```yaml
router:
  llama_server: /path/to/rie-server
  models:
    - name: fast
      file: ~/models/qwen2.5-1.5b-instruct-q4_0.gguf
      extra_args: [--auto-reward, --adapter, ~/.stitap/fast.riea]
```

## Teaching it explicitly

Every response carries an `id` (`chatcmpl-…`). The server keeps the last 16 completions.

```bash
# reward or penalize a whole reply, its tool call only, or its text only
curl -s localhost:8080/v1/feedback -d '{"id":"chatcmpl-…","reward":-1,"scope":"tool_call"}'

# show the right answer: the first wrong token is pushed down, the correction pushed up
curl -s localhost:8080/v1/corrections -d '{"id":"chatcmpl-…","tool_call":{"name":"fin.loan_emi","arguments":{"principal":1000000,"annualRate":8.5,"months":240}}}'
curl -s localhost:8080/v1/corrections -d '{"id":"chatcmpl-…","content":"The EMI is ₹8,678 per month."}'

curl -s localhost:8080/v1/rl            # experts, updates, mean reward
curl -s -X POST localhost:8080/v1/rl/reset
```

## How the learning works

The final hidden state `h` of each generated token goes through a router
`g = softmax(W_g ĥ)` over K experts (default 4). Each expert is a rank-r (default 8)
map, and their weighted sum is added to the logits: `Σ_k g_k A_k B_k ĥ`, clamped to
±8. Different kinds of request land in different parts of hidden-state space, so the
experts specialize, for example on finance tools versus file tools. A reward `R` for a
token updates the experts with the REINFORCE gradient `R·(onehot − p)` over the top-32
candidates recorded at sampling time. The update is closed-form and touches only those
vocabulary rows, so it takes milliseconds on a phone.

Limits: learning needs the CPU backend (it reads hidden states), and only the last 16
completions can receive feedback. Corrections act on the output layer, so they steer
choices the model already considers plausible. They do not teach new facts.

## Checking the integration

`scripts/harness-check.mjs` reproduces the harness's request and streaming format
(`tools` + `tool_choice: auto`, `stream_options.include_usage`, tool-call deltas merged by
index, tool results sent back) and runs a short agent loop: a forced tool call, an error
result that triggers auto-reward, then feedback and a correction. CI runs it on every push.

## Sandboxed end-to-end test with the real harness

`scripts/sandbox-harness.sh HARNESS_DIR MODEL.gguf [harness args]` runs the real stitaP
harness against `rie-server` inside Linux namespaces:
- no internet: a private network with loopback only, and `rie-server` runs inside it;
- the harness checkout is mounted read-only;
- a throwaway home directory, with approvals set to `deny`.

```bash
scripts/sandbox-harness.sh ~/tool-harness model.gguf doctor
SERVER_ARGS="--tool-choice required" scripts/sandbox-harness.sh ~/tool-harness model.gguf \
    batch prompts.jsonl --out /tmp/traj.jsonl
```

`requests.jsonl` in `$WORK` logs every request together with what the server generated
(set `RIE_SERVER_LOG=file` to get the same log from any server).

What running the real harness this way uncovered and fixed:

| Finding | Fix |
|---|---|
| The harness's `use_tool` meta-tool takes a free-form `arguments` object. A small model kept adding keys until it hit `max_tokens`, and the harness received an empty reply | A size budget per call (`--max-call`, default 16 KB): past it the call *winds down*, meaning strings close, no new keys or items are added (except missing required ones) and containers close, always ending in a valid call. Free-form object keys are now length-capped too |
| Auto-reward on long tool calls (100–300 tokens, repeated) made the correction experts diverge (weights up to 1e19, then NaN) | The learning rate is scaled by 1/√length for trajectory rewards, every update is clipped, A is bounded, B rows keep their norm, router rows are norm-capped, non-finite updates are rejected, and adapter files with non-finite weights are refused. A 300-step × 20-round stress test is part of the suite. Corrections stay full strength |
| `write_file` was executed despite `approvals: deny` | Harness behaviour (deny seems to cover only some tools). Keep the checkout read-only, or run in a sandbox, when testing |

With a random-weight test model the harness accepts every call `rie-server` produces
(its own validation passes), tool results come back, and auto-reward learns from the
failures. Judging the agent's actual task success needs a real instruct model (for
example Qwen2.5-1.5B/3B-Instruct), which could not be downloaded in the environment where
this was developed.
