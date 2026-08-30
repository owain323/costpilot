"""A mock real-world app for scanner tests (intentionally contains multiple LLM call patterns)."""

import openai
from anthropic import Anthropic
from langchain_openai import ChatOpenAI

client = openai.OpenAI()
MODEL_NAME = "gpt-4o"


def classify(text: str) -> str:
    """Direct OpenAI SDK call (constant model)."""
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": text}],
        max_tokens=100,
    )
    return resp.choices[0].message.content


def summarize(text: str) -> str:
    """OpenAI SDK call (model is a variable -> model=None, no guessing)."""
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": text}],
    )
    return resp.choices[0].message.content


def ask_anthropic(text: str) -> str:
    """Direct Anthropic SDK call."""
    aclient = Anthropic()
    resp = aclient.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=200,
        messages=[{"role": "user", "content": text}],
    )
    return resp.content[0].text


def use_langchain(text: str) -> str:
    """LangChain ChatOpenAI constructor."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    return llm.invoke(text).content


def helper(x: int) -> int:
    """Plain function: must NOT be detected as an LLM call."""
    return x * 2
