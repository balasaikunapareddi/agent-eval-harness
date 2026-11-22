"""Prompt text used by the reference agent's live backend, in one place.

Note: the *system prompts under test* are not here — they live in the
agent config files (``configs/*.yaml``), because the config is the demo's
subject matter: editing it is exactly what the harness detects. What
lives here is the fixed instruction scaffolding the live backend wraps
around whatever config it is given:

- ``ACTION_INSTRUCTIONS`` — the JSON action protocol appended to every
  rendered prompt (see ``agent.LiteLLMBackend.next_action``).
- ``INVALID_JSON_RETRY`` — suffix appended when the model replies with
  unparseable output.
- ``JUDGE_INSTRUCTIONS`` / ``JUDGE_REPLY_FORMAT`` — the LLM-judge prompt
  frame (see ``agent.LiteLLMBackend.judge``).
"""

ACTION_INSTRUCTIONS = (
    "Reply with exactly ONE JSON object, nothing else. Either:\n"
    '{"action": "tool_call", "tool": "<tool name>", "args": {...}}\n'
    "or, when you have everything you need:\n"
    '{"action": "final", "answer": "<your answer to the user>"}'
)

INVALID_JSON_RETRY = "\n\nYour last reply was not valid JSON. Reply with ONE JSON object only."

JUDGE_INSTRUCTIONS = "You are grading an AI assistant's answer against a rubric.\n"

JUDGE_REPLY_FORMAT = 'Reply with exactly ONE JSON object: {"score": <0.0-1.0>, "reason": "<short reason>"}'
