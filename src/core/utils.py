"""
Lab 11 — Helper Utilities
"""
from core.config import get_llm_provider, PROVIDER_OPENROUTER  # noqa: F401
from core.openai_runtime import OpenAIRunner


async def chat_with_agent(agent, runner, user_message: str, session_id=None):
    """Send a message to the agent and get the response.

    Works with OpenAIRunner (OpenAI Red / OpenRouter Blue) and Google ADK (Gemini Red).
    """
    provider = getattr(runner, "provider", None)
    if isinstance(runner, OpenAIRunner) or provider in ("openrouter", "openai"):
        text = await runner.chat(agent, user_message)
        return text, None

    from google.genai import types

    user_id = "student"
    app_name = runner.app_name

    session = None
    if session_id is not None:
        try:
            session = await runner.session_service.get_session(
                app_name=app_name, user_id=user_id, session_id=session_id
            )
        except (ValueError, KeyError):
            pass

    if session is None:
        try:
            session = await runner.session_service.create_session(
                app_name=app_name, user_id=user_id
            )
        except Exception:
            session = await runner.session_service.create_session(
                app_name=app_name, user_id=user_id
            )

    content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=user_message)],
    )

    # Check input plugins before calling model to prevent quota waste on blocked inputs
    plugins = getattr(runner, "plugins", None)
    if not plugins and hasattr(runner, "plugin_manager"):
        plugins = getattr(runner.plugin_manager, "plugins", []) or []
    for plugin in plugins:
        cb = getattr(plugin, "on_user_message_callback", None)
        if cb:
            class _Ctx:
                user_id = "student"
            import inspect
            if inspect.iscoroutinefunction(cb):
                res = await cb(invocation_context=_Ctx(), user_message=content)
            else:
                res = cb(invocation_context=_Ctx(), user_message=content)
            if res is not None:
                text = ""
                if hasattr(res, "parts") and res.parts:
                    for p in res.parts:
                        if hasattr(p, "text") and p.text:
                            text += p.text
                return text or "I cannot process that request. I only help with VinBank banking questions.", None

    import asyncio

    for attempt in range(3):
        try:
            final_response = ""
            async for event in runner.run_async(
                user_id=user_id, session_id=session.id, new_message=content
            ):
                if hasattr(event, "content") and event.content and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, "text") and part.text:
                            final_response += part.text
            return final_response, session
        except Exception as e:
            err_msg = str(e)
            if attempt < 2 and any(code in err_msg for code in ("503", "429", "UNAVAILABLE", "ResourceExhausted", "high demand")):
                await asyncio.sleep(2 * (attempt + 1))
                continue
            raise
