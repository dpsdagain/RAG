"""Chainlit conversational frontend for the RAG system."""
import json
import httpx
import httpx_sse
import chainlit as cl

API_BASE_URL = "http://localhost:8000/v1"

@cl.on_chat_start
async def on_chat_start():
    """Initialize the chat session."""
    cl.user_session.set("conversation_id", None)
    await cl.Message(
        content="Welcome to the RAG System! Ask a question or upload a document (PDF, code, text) to get started."
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming user messages."""
    # 1. Handle file uploads if any
    if message.elements:
        for element in message.elements:
            if isinstance(element, cl.File):
                await handle_file_upload(element)
    
    # If there's no text message, just return
    if not message.content.strip():
        return

    # 2. Process text query
    msg = cl.Message(content="")
    await msg.send()
    
    conversation_id = cl.user_session.get("conversation_id")
    
    request_payload = {
        "query": message.content,
        "conversation_id": conversation_id,
        "stream": True
    }
    
    sources: list[dict] = []
    sources_rendered = False

    async def _render_sources_now() -> None:
        """Attach the side-panel citations as soon as we have them."""
        nonlocal sources_rendered
        if sources_rendered or not sources:
            return
        elements = []
        for i, source in enumerate(sources):
            content = f"**Score:** {source.get('score', 0):.2f}\n\n{source.get('content_snippet', '')}"
            name = f"[{i+1}] {source.get('source_uri', 'Unknown').split('/')[-1]}"
            elements.append(cl.Text(name=name, content=content, display="side"))
        msg.elements = elements
        await msg.update()
        sources_rendered = True

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with httpx_sse.aconnect_sse(
                client,
                "POST",
                f"{API_BASE_URL}/chat/completions",
                json=request_payload
            ) as event_source:
                async for sse in event_source.aiter_sse():
                    data = json.loads(sse.data)

                    if sse.event == "sources":
                        # Render the side panel BEFORE tokens stream so
                        # citations are visible alongside the answer.
                        sources = list(data.get("sources", []))
                        await _render_sources_now()

                    elif sse.event == "token":
                        await msg.stream_token(data)

                    elif sse.event == "done":
                        new_conv_id = data.get("conversation_id")
                        if new_conv_id:
                            cl.user_session.set("conversation_id", new_conv_id)

                    elif sse.event == "error":
                        msg.content = f"❌ Error: {data}"
                        await msg.update()
                        return

    except Exception as e:
        msg.content = f"❌ Connection to backend failed: {str(e)}"
        await msg.update()
        return

    # Fallback: if sources arrived after tokens (cached/older paths) ensure
    # they still get rendered before the final update.
    await _render_sources_now()

    if sources:
        refs = ", ".join(
            f"[{i+1}] {s.get('source_uri', 'Unknown').split('/')[-1]}"
            for i, s in enumerate(sources)
        )
        await msg.stream_token(f"\n\n**Sources:** {refs}")

    await msg.update()

async def handle_file_upload(file_element: cl.File):
    """Send uploaded file to the ingestion API."""
    processing_msg = cl.Message(content=f"⏳ Ingesting document: `{file_element.name}`...")
    await processing_msg.send()
    
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            with open(file_element.path, "rb") as f:
                files = {"file": (file_element.name, f, "application/octet-stream")}
                response = await client.post(f"{API_BASE_URL}/ingest/file", files=files)
                
            if response.status_code == 200:
                data = response.json()
                chunks = data.get("chunks_created", 0)
                processing_msg.content = f"✅ Successfully ingested `{file_element.name}`! Created {chunks} searchable chunks."
            else:
                processing_msg.content = f"❌ Failed to ingest `{file_element.name}`: {response.text}"
    except Exception as e:
        processing_msg.content = f"❌ Error uploading `{file_element.name}`: {str(e)}"
        
    await processing_msg.update()
