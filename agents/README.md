# Python Agent

Stateful agent with tool execution and event streaming. Built on `ai_models`.

## Usage

```python
from ai_models import get_model, get_env_api_key
from agent import Agent

model = get_model("openrouter", "openai/gpt-4o-mini", api_key=get_env_api_key("openrouter"))
agent = Agent(system_prompt="You are concise.", model=model)

agent.subscribe(lambda ev: (
    print(ev.get("assistant_message_event", {}).get("delta", ""), end="", flush=True)
    if ev.get("type") == "message_update" else None
))

await agent.prompt("Hello!")
```

## Concepts

- **Agent**: Holds `system_prompt`, `model`, `tools`, `messages`. Call `prompt(text)` to add a user message and run the loop (stream assistant, execute tools, repeat until no tool calls).
- **Events**: Subscribe with `agent.subscribe(callback)`. Events include `agent_start`, `turn_start`, `message_start` / `message_update` / `message_end`, `tool_execution_start` / `tool_execution_end`, `turn_end`, `agent_end`.
- **Tools**: Implement the `AgentTool` protocol: `name`, `description`, `parameters` (JSON Schema), and `async execute(tool_call_id, params) -> AgentToolResult`. Set with `agent.set_tools([...])`.

## API

- `Agent(system_prompt=..., model=..., tools=None, messages=None)`
- `agent.prompt(text, options=None)` – async, returns updated `messages`
- `agent.subscribe(callback)` – returns unsubscribe function
- `agent.set_system_prompt`, `agent.set_model`, `agent.set_tools`, `agent.replace_messages`, `agent.clear_messages`, `agent.reset`
- `agent_loop(prompts, context, model, options)` – low-level async generator of events (used by `Agent`)

## Example with a tool

See `examples/agent_demo.py` for a minimal run. To add a tool:

```python
class EchoTool:
    name = "echo"
    description = "Echo the given message"
    parameters = {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]}

    async def execute(self, tool_call_id: str, params: dict) -> dict:
        return {"content": [{"type": "text", "text": params.get("message", "")}], "details": {}}

agent.set_tools([EchoTool()])
await agent.prompt("Use the echo tool to say hi.")
```
