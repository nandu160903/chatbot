"""
Chainlit + Hugging Face Chatbot Application (Python Script Version)
--------------------------------------------------------------------
This script integrates Chainlit with Hugging Face Transformers for AI chat.
Uses Qwen2.5-0.5B-Instruct - small, efficient, and multi-purpose model.
"""

# 1️⃣ Importing Required Libraries
import os
import asyncio
from dotenv import load_dotenv
from typing import Dict, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

import chainlit as cl
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.types import ThreadDict

# 2️⃣ Load Environment Variables
load_dotenv()

# NOTE: These environment variables are hardcoded for demonstration.
# In production, keep them in your .env file for better security.
os.environ['CHAINLIT_AUTH_SECRET'] = "r>>aPxK9Iwl%KMjr,sjeIoP@I.kGOLb*kwriPYwtW$S9vJVR2HYFh.JUc_0J:PF."
os.environ['DATABASE_URL'] = "postgresql+asyncpg://myuser:mypassword@localhost:5432/chatbot"

# 3️⃣ Hugging Face Model Initialization

# Model selection: Qwen2.5-0.5B-Instruct - lightweight, efficient, multi-purpose
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔧 Using device: {device}")

# Load model and tokenizer
print(f"📦 Loading model: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto" if device == "cuda" else None,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    trust_remote_code=True
)

if device == "cpu":
    model = model.to(device)

print("✅ Model loaded successfully!")

# 4️⃣ Authentication Callback

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    # Simple hardcoded authentication (replace with secure check in production)
    if username == "admin" and password == "chainlit":
        return cl.User(identifier=username)
    return None

# 5️⃣ Chat Session Initialization

@cl.on_chat_start
async def on_chat_start():
    """
    Called when a new chat session starts. Initializes chat history.
    """
    cl.user_session.set('chat_history', [])
    await cl.Message(
        content=f"👋 Hello! Welcome to the AI chatbot powered by **{MODEL_NAME}**.\nI'm a lightweight, multi-purpose assistant ready to help you!"
    ).send()

# 6️⃣ Database Layer for Chat History Persistence

@cl.data_layer
def get_data_layer():
    """
    Establish SQLAlchemy-based data layer for persisting chat history into PostgreSQL.
    """
    return SQLAlchemyDataLayer(conninfo=os.getenv("DATABASE_URL"))

# 7️⃣ Resume Chat Session

@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    """
    Reload previous chat history when resuming a chat thread.
    """
    cl.user_session.set("chat_history", [])
    
    for message in thread['steps']:
        if message['type'] == 'user_message':
            cl.user_session.get("chat_history").append(
                {'role':'user', 'content': message['output']}
            )
        elif message['type'] == 'assistant_message':
            cl.user_session.get("chat_history").append(
                {'role':'assistant', 'content': message['output']}
            )

# 8️⃣ Chat Message Handling Logic

@cl.on_message
async def on_message(message: cl.Message):
    """
    Main chat handler which takes incoming user message, forwards it to Hugging Face LLM, 
    streams back response, and updates chat history.
    """
    chat_history = cl.user_session.get("chat_history")

    # Add user message to history
    chat_history.append({'role':'user', 'content':message.content})

    # Create response message
    cb = cl.Message(content="")
    await cb.send()

    # Prepare conversation for the model
    conversation = []
    for msg_item in chat_history:
        conversation.append({"role": msg_item["role"], "content": msg_item["content"]})

    # Tokenize input
    inputs = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(device)

    # Setup streaming
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    generation_kwargs = {
        "input_ids": inputs,
        "max_new_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "streamer": streamer
    }

    # Run generation in a separate thread for streaming
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # Stream the response
    assistant_response = ''
    for new_text in streamer:
        assistant_response += new_text
        await cb.stream_token(new_text)

    # Wait for generation to complete
    thread.join()

    # Add assistant response to history
    chat_history.append({'role':'assistant', 'content':assistant_response})

    await cb.update()

# ✅ Application Setup Completed
