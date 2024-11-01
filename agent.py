from __future__ import annotations
from typing import Annotated

import logging
from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.multimodal import MultimodalAgent
from livekit.plugins import openai


load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("my-worker")
logger.setLevel(logging.INFO)


async def entrypoint(ctx: JobContext):
    fnc_ctx = llm.FunctionContext()

    @fnc_ctx.ai_callable()
    async def get_weather(
        location: Annotated[
            str, llm.TypeInfo(description="The location to get the weather for.")
        ],
    ):
        """Called when the user asks about the weather. This function will return the weather for the given location."""
        logger.info(f"getting weather for {location}")
        return f"The weather in {location} is sunny and 50C."

    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    
    participant = await ctx.wait_for_participant()

    run_multimodal_agent(ctx, participant, fnc_ctx)

    logger.info("agent started")


def run_multimodal_agent(ctx: JobContext, participant: rtc.Participant, fnc_ctx: llm.FunctionContext):
    logger.info("starting multimodal agent")

    model = openai.realtime.RealtimeModel(
        instructions=(
            "Your are an assistant named 'Steve' who is a helpful and friendly AI assistant."
        ),
        modalities=["audio", "text"],
    )
    assistant = MultimodalAgent(model=model, fnc_ctx=fnc_ctx)
    assistant.start(ctx.room, participant)

    session = model.sessions[0]
    session.conversation.item.create(
        llm.ChatMessage(
            role="assistant",
            content="Please begin the interaction with the user in a manner consistent with your instructions.",
        )
    )
    session.response.create()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
        )
    )
