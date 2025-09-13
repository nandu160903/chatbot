import chainlit as cl
from history_db import init_db, save_message, get_chat_history
import uuid

init_db()

@cl.on_chat_start
async def on_chat_start():
    user = cl.user_session.user
    if not user:
        await cl.Message(content="⚠️ You are not logged in.").send()
        return

    session_id = str(uuid.uuid4())
    cl.user_session.set("session_id", session_id)
    await cl.Message(content=f"👋 Hello, **{user.username}**!\nType anything or `/history` to view your past chats.").send()

@cl.on_message
async def on_message(message: cl.Message):
    user = cl.user_session.user
    session_id = cl.user_session.get("session_id")

    if message.content.strip().lower() == "/history":
        history = get_chat_history(user.id)
        if not history:
            await cl.Message(content="No previous history found.").send()
        else:
            formatted = "\n\n".join(
                [f"**{role.capitalize()}** ({timestamp}):\n{msg}" for role, msg, timestamp in history]
            )
            await cl.Message(content=formatted).send()
        return

    save_message(session_id, user.id, "user", message.content)

    response = f"Echo: {message.content}"
    await cl.Message(content=response).send()

    save_message(session_id, user.id, "assistant", response)
