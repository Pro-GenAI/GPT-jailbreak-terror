from dotenv import load_dotenv
from IPython.display import display, Markdown
import openai
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

import os


def display_md(md: str | None):
    if md:
        display(Markdown(md))


load_dotenv(override=True)

client = openai.OpenAI()
model = os.getenv("OPENAI_MODEL", "")
if not model:
    raise ValueError("OPENAI_MODEL environment variable not set")


def base_message(content: str, role: str) -> ChatCompletionMessageParam:
    return {"role": role, "content": content}  # type: ignore


def user_message(content: str) -> ChatCompletionMessageParam:
    return base_message(content, "user")


def system_message(content: str) -> ChatCompletionMessageParam:
    return base_message(content, "system")


def bot_message(content: str) -> ChatCompletionMessageParam:
    return base_message(content, "assistant")


def get_response(
    messages: str | list[ChatCompletionMessageParam],
    **kwargs
) -> str | None:
    if isinstance(messages, str):
        messages = [user_message(messages)]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs,  # More arguments like seed, temperature, etc.
    )
    return response.choices[0].message.content

