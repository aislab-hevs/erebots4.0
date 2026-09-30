# Running a model

Erebots4.0 needs an OpenAI-compatible LLM endpoint. There is no bundled model and
no offline mode: the orchestrator routes with a model, specialists answer with
one, and without an endpoint every turn fails at the first call.

Two requirements, whatever you run:

- **Tool calling.** Specialists reach their MCP tools through the model's tool
  interface. A model or server without it will answer, but never call a tool.
- **Enough context** for the system prompt, the routing prompt listing every
  agent and its skills, and up to `chat_history_max_messages` turns.

Anything OpenAI-compatible works — llama.cpp, vLLM, Ollama, or a hosted API.
Below is llama.cpp, which is what this project is developed against.

## Build llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON          # drop -DGGML_CUDA=ON without an NVIDIA GPU
cmake --build build --config Release -j
```

`-DGGML_CUDA=ON` needs the CUDA toolkit and a current driver. Without a GPU the
build still works and the model runs on CPU — slower, but complete.

## Serve a model

### On a laptop

A 4B model is enough to exercise the whole path: routing, A2A, MCP tool calls
and a composed answer. Expect it to route less reliably than a large one, and to
be slow on CPU.

```bash
./build/bin/llama-server \
    -hf unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL \
    --alias local \
    --host 127.0.0.1 --port 8080 \
    --ctx-size 16384 \
    --jinja
```

About 2.5 GB, downloaded on first run.

**`--jinja` is not optional.** It makes llama.cpp apply the model's own chat
template, which is what emits and parses tool calls. Without it the server
accepts a `tools` array and silently never uses one — the agents will answer,
and never call a tool.

### On a workstation with a GPU

With around 20 GB of VRAM a much larger mixture-of-experts model fits, which is
what the published results were produced with:

```bash
./build/bin/llama-server \
    -hf unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ4_XS \
    --alias Qwen3.6-35B-A3B \
    --host 127.0.0.1 --port 8080 \
    --ctx-size 131072 \
    --n-gpu-layers 99 \
    --flash-attn on \
    --cache-type-k q4_0 --cache-type-v q4_0 \
    --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 \
    --jinja
```

The quantised KV cache (`--cache-type-k/v q4_0`) is what makes a 128K context fit
alongside the weights. Drop `--ctx-size` before dropping the quantisation if you
run short of memory.

If the server runs on another machine, tunnel it rather than exposing it:

```bash
ssh -NL 8080:localhost:8080 USER@WORKSTATION
```

## Check it answers

```bash
curl -s http://127.0.0.1:8080/v1/models | jq '.data[].id'
```

## Point Erebots4.0 at it

Every agent has its own endpoint setting, so they can run on different servers.
Pointing them all at one is the normal local case — see
[configuration.md](configuration.md) for the full list.

```bash
LLM=http://127.0.0.1:8080/v1
MODEL=local          # the --alias above, or the id printed by /v1/models

export orchestrator_api_base_url=$LLM conversion_api_base_url=$LLM personal_api_base_url=$LLM
export orchestrator_model=$MODEL conversion_model=$MODEL personal_model=$MODEL
```

The README quickstart uses exactly these variables. If your server listens on
another port, change `LLM` and nothing else.

## Check the whole path, not just the model

A model that answers is not yet proof that tools work. The conversion agent is
the shortest test, because its answer is wrong unless a tool really ran:

```bash
echo "how many pounds is 70 kg?" | uv run python examples/cli_chat.py
```

Expect `70 kg = 154.324 lb`. A slightly different number means the model
computed it itself instead of calling the tool — check `--jinja`, and see
[adding-a-tool.md](adding-a-tool.md).

## If something is wrong

| Symptom | Cause |
|---|---|
| `APIConnectionError: Connection error` | Nothing is listening on that port; check `*_api_base_url`. |
| Every turn is answered by the orchestrator | Routing is failing. Set `orchestrator_thinking=true` to see its reasoning, or try a larger model. |
| Answers are right but no tool is ever called | The server is not applying the chat template — add `--jinja`. |
| Long pauses, then a reply | Normal on CPU. One turn can take minutes with a large model. |
