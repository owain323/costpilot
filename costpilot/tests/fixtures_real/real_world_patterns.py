"""Real-world invocation patterns found in open-source repos during the P1.5 benchmark.

Covers patterns the original rules missed plus model-resolution sources (P2a):
1. Split invocation: assign a `*.chat.completions` object to a variable, call it later.
2. Beta API: `client.beta.chat.completions.parse(...)`.
3. Model from module constant / function param default / env fallback.
4. Unresolvable model (must stay unknown, never guessed).
"""

import os

import openai
from anthropic import Anthropic

DEFAULT_MODEL = "gpt-4o-mini"


class ClientWrapper:
    def __init__(self) -> None:
        # split-invocation pattern: chat.completions object stored, called later
        self.client = openai.OpenAI(api_key="sk-test").chat.completions

    def run(self, text: str) -> str:
        response = self.client.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": text}],
        )
        return response.choices[0].message.content

    def run_structured(self, text: str) -> str:
        completion = self.client.parse(
            model="gpt-4o",
            messages=[{"role": "user", "content": text}],
        )
        return str(completion)


# module-level split invocation (name form, not attribute form)
fallback_client = openai.AzureOpenAI(api_key="sk-azure").chat.completions


def fallback(text: str) -> str:
    return (
        fallback_client.create(model="gpt-4o", messages=[{"role": "user", "content": text}])
        .choices[0]
        .message.content
    )


def structured(text: str) -> str:
    client = openai.OpenAI()
    completion = client.beta.chat.completions.parse(
        model="gpt-4o", messages=[{"role": "user", "content": text}]
    )
    return str(completion)


def from_module_constant(text: str) -> str:
    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": text}],
    )
    return resp.choices[0].message.content


def from_env_default(text: str) -> str:
    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model=os.getenv("MODEL", "gpt-4o"),
        messages=[{"role": "user", "content": text}],
    )
    return resp.choices[0].message.content


def from_param_default(text: str, model: str = "claude-haiku-3-5") -> str:
    aclient = Anthropic()
    resp = aclient.messages.create(
        model=model,
        max_tokens=200,
        messages=[{"role": "user", "content": text}],
    )
    return resp.content[0].text


def unresolvable_model(text: str) -> str:
    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model=pick_model(),  # cannot resolve statically: must stay unknown
        messages=[{"role": "user", "content": text}],
    )
    return resp.choices[0].message.content


def pick_model() -> str:
    return "gpt-4o"
