import torch

from transformers import AutoTokenizer

model_path = "Qwen/Qwen3-4B-Instruct-2507"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

tools = [{
    "name": "code_interpreter",
    "description": "A tool for executing code.",
    "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
}]

messages = [
    {"role": "system", "content": "You are a helpful assistant..."},
    {"role": "user", "content": "Solve the following problem..."},
    {
        "role": "assistant",
        "content": "Let me compute.",
        "tool_calls": [{
            "type": "function",
            "id": "call_0",
            "function": {"name": "code_interpreter", "arguments": {"code": "print(1+1)"}},
        }],
    },
    {"role": "tool", "content": "2", "tool_call_id": "call_0"},
    {"role": "assistant", "content": "Answer: 2"},
]

rendered = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=False, tokenize=False)
print("=== Full Rendered Prompt ===")
print(tokenizer.decode(tokenizer.encode(rendered), skip_special_tokens=True))

print("\n=== Individual Turns ===")
for msg in messages:
    print(msg["role"], ":", msg["content"])