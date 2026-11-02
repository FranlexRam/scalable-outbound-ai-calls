import asyncio
import json
import logging
import os

from dotenv import load_dotenv
from livekit import api
from livekit.protocol.sip import SIP_MEDIA_ENCRYPT_DISABLE
from livekit.agents import (
    AgentSession,
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
)
from livekit.agents.inference import TTS as InferenceTTS, STT as InferenceSTT
from livekit.agents.voice import Agent
from livekit.agents.voice.room_io import RoomInputOptions
from livekit.plugins import openai, silero

load_dotenv()

logger = logging.getLogger("outbound-agent")
logging.basicConfig(level=logging.INFO)

AGENT_NAME = "outbound-agent"
SIP_TRUNK_ID = os.environ["LIVEKIT_SIP_TRUNK_ID"]
DEFAULT_TTS_MODEL = "cartesia/sonic-2"


async def entrypoint(ctx: JobContext):
    metadata: dict = {}
    if ctx.job.metadata:
        try:
            metadata = json.loads(ctx.job.metadata)
        except json.JSONDecodeError:
            logger.warning("Could not parse job metadata as JSON")

    phone_number: str = metadata.get("phone_number", "")
    agent_instructions: str = metadata.get(
        "agent_instructions",
        "You are a friendly assistant making a courtesy call. Be warm, concise, and helpful.",
    )
    agent_display_name: str = metadata.get("agent_name", "AI Assistant")
    voice_id: str | None = metadata.get("voice_id") or None
    language: str = metadata.get("language") or "en"
    contact_name: str = metadata.get("contact_name", "")
    personalization_context: str = metadata.get("personalization_context", "")

    if not phone_number:
        logger.error("No phone_number in job metadata — aborting")
        return

    if language == "es":
        agent_instructions = "Responde siempre en español.\n\n" + agent_instructions

    if contact_name:
        agent_instructions = (
            agent_instructions
            + f"\n\nThe person you are calling is named {contact_name}. Address them by name."
        )

    if personalization_context:
        agent_instructions = (
            agent_instructions
            + "\n\nAdditional context about the person you're calling:\n"
            + personalization_context
        )

    logger.info(
        "Starting outbound call to %s as '%s' (voice=%s, language=%s)",
        phone_number, agent_display_name, voice_id or DEFAULT_TTS_MODEL, language,
    )

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    if voice_id:
        tts = InferenceTTS.from_model_string(voice_id)
    else:
        tts = InferenceTTS(model=DEFAULT_TTS_MODEL)

    # Dial out via SIP trunk before starting the agent session
    lkapi = api.LiveKitAPI(
        url=os.environ["LIVEKIT_URL"],
        api_key=os.environ["LIVEKIT_API_KEY"],
        api_secret=os.environ["LIVEKIT_API_SECRET"],
    )
    try:
        await lkapi.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                sip_trunk_id=SIP_TRUNK_ID,
                sip_call_to=phone_number,
                room_name=ctx.room.name,
                participant_identity=f"sip_{phone_number}",
                participant_name=f"Callee {phone_number}",
                wait_until_answered=True,
                media_encryption=SIP_MEDIA_ENCRYPT_DISABLE,
            )
        )
    except Exception as exc:
        logger.error("Failed to create SIP participant: %s", exc)
        return
    finally:
        await lkapi.aclose()

    logger.info("SIP participant created — waiting for them to join the room…")
    sip_participant = await _wait_for_sip_participant(ctx)

    if sip_participant is None:
        logger.error("SIP participant never joined — ending session")
        return

    logger.info("SIP participant joined: %s — starting agent session", sip_participant.identity)

    if language == "es":
        stt = InferenceSTT(model="deepgram/nova-2", language="es")
    else:
        stt = InferenceSTT(model="deepgram/nova-2-phonecall")

    agent = Agent(
        instructions=agent_instructions,
        stt=stt,
        llm=openai.LLM(
            model="anthropic/claude-haiku-5.5",
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENAI_API_KEY")
        ),
        tts=tts,
        vad=silero.VAD.load(),
    )

    session = AgentSession()
    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            participant_identity=sip_participant.identity,
        ),
    )

    greeting = f"Greet the person warmly. Your name is {agent_display_name}."
    if contact_name:
        greeting += f" You are calling {contact_name} — address them by name in your greeting."
    await session.generate_reply(instructions=greeting)


async def _wait_for_sip_participant(ctx: JobContext, timeout: float = 60.0):
    for p in ctx.room.remote_participants.values():
        if p.identity.startswith("sip_"):
            return p

    future: asyncio.Future = asyncio.get_event_loop().create_future()

    def on_participant_connected(participant):
        if participant.identity.startswith("sip_") and not future.done():
            future.set_result(participant)

    ctx.room.on("participant_connected", on_participant_connected)

    try:
        return await asyncio.wait_for(future, timeout=timeout)
    except asyncio.TimeoutError:
        logger.error("Timed out waiting for SIP participant to join")
        return None
    finally:
        ctx.room.off("participant_connected", on_participant_connected)


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name=AGENT_NAME,
        )
    )