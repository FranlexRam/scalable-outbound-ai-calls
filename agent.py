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
    llm,
)
from livekit.agents.inference import TTS as InferenceTTS, STT as InferenceSTT
from livekit.agents.voice import Agent
from livekit.agents.voice.room_io import RoomInputOptions
from livekit.plugins import openai, silero

load_dotenv(override=True)

logger = logging.getLogger("outbound-agent")
logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("livekit").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

AGENT_NAME = "outbound-agent"
SIP_TRUNK_ID = os.environ.get("LIVEKIT_SIP_TRUNK_ID", "")
DEFAULT_TTS_MODEL = "cartesia/sonic-2"


async def entrypoint(ctx: JobContext):
    load_dotenv(override=True)

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
        logger.info("No phone_number in metadata. Assuming INBOUND call.")
        agent_instructions = "You are a helpful AI receptionist for Plexus Code. Greet the caller warmly and assist them with their inquiries."
        agent_display_name = "Plexus Voice Assistant"

    if language == "es":
        # 1. Identidad y contexto 100% en Español
        agent_instructions = f"Tu identidad: Eres {agent_display_name}.\n\n" + agent_instructions

        if contact_name:
            agent_instructions += (
                f"\n\nLa persona a la que llamas se llama {contact_name}. "
                "Trátala de tú con cercanía, pero no repitas su nombre en todas las frases."
            )

        if personalization_context:
            agent_instructions += (
                "\n\nContexto sobre la llamada y el cliente:\n"
                + personalization_context
            )

        # 2. Motor de Naturalidad Fonética y Conversacional (Español)
        agent_instructions += (
            "\n\nREGLAS ESTRICTAS DE REALISMO HUMANO Y FONÉTICA (OBLIGATORIO):\n"
            "1. IDIOMA Y TONO: Habla siempre en español neutro latinoamericano, con tono cálido, relajado y espontáneo, como una persona real llamando desde su oficina. Jamás suenes a locutor comercial.\n"
            "2. PROHIBIDO SIGNOS DE EXCLAMACIÓN: NUNCA uses signos de admiración (¡ !). El motor de voz los exagera y suena robótico. Usa únicamente puntos, comas, puntos suspensivos (...) y signos de interrogación (?).\n"
            "3. ESCRITURA FONÉTICA: Escribe TODOS los números, porcentajes y horas en letras minúsculas exactamente como se pronuncian (ejemplo: escribe 'cincuenta por ciento' en lugar de '50%', 'diez de la mañana' en lugar de '10 am').\n"
            "4. MULETILLAS FLUIDAS AL INICIAR: Inicia tus respuestas con conectores naturales seguidos de coma como 'Claro, mira,', 'Ah, entiendo,', 'Sí, te cuento rapidito,', 'Bueno, fíjate que'. NUNCA pongas puntos suspensivos (...) en las primeras palabras para no retrasar el inicio de tu voz.\n"
            "5. DOSIFICACIÓN DE INFORMACIÓN: NUNCA sueltes todo el discurso de golpe. Habla en turnos muy breves de 1 sola oración (máximo 15 a 22 palabras) y termina con una pregunta corta y natural para escuchar al cliente.\n"
            "6. FLUJO DESPUÉS DEL ALÓ: Cuando el cliente responda a tu '¿Aló?', preséntate brevemente, menciona el motivo en una frase sencilla y pregúntale si tiene un minutito."
        )
    else:
        # 1. Identidad y contexto 100% en Inglés
        agent_instructions = f"Your identity: You are {agent_display_name}.\n\n" + agent_instructions

        if contact_name:
            agent_instructions += (
                f"\n\nThe person you are calling is named {contact_name}. "
                "Be warm and conversational, but do not overuse their name in every sentence."
            )

        if personalization_context:
            agent_instructions += (
                "\n\nAdditional context about the call and the person:\n"
                + personalization_context
            )

        # 2. Motor de Naturalidad Fonética y Conversacional (Inglés)
        agent_instructions += (
            "\n\nSTRICT HUMAN REALISM AND PHONETIC RULES (MANDATORY):\n"
            "1. TONE AND DELIVERY: Speak in a warm, relaxed, spontaneous tone, like a real person calling from a quiet office. Never sound like a scripted telemarketer or radio announcer.\n"
            "2. NO EXCLAMATION MARKS: NEVER use exclamation marks (!). The TTS engine over-acts them and sounds robotic. Use only periods, commas, ellipses (...), and question marks (?).\n"
            "3. PHONETIC SPELLING: Spell out ALL numbers, percentages, and times in lowercase words exactly as spoken (e.g., write 'fifty percent' instead of '50%', 'ten in the morning' instead of '10 am').\n"
            "4. FLUENT FILLERS AT START: Start responses with natural connectors followed by a comma like 'Yeah, so,', 'Oh, got it,', 'Sure, real quick,'. NEVER put ellipses (...) in the first few words so your voice starts immediately.\n"
            "5. PACING AND TURN LENGTH: NEVER deliver a long pitch all at once. Keep every turn to 1 short sentence (15 to 22 words max) and end with a natural, low-pressure question.\n"
            "6. POST-HELLO FLOW: Once the person replies to your initial 'Hello?', introduce yourself briefly, mention the main reason for calling in one simple phrase, and ask if caught them at a good time."
        )

    logger.info(
        "Starting call as '%s' (voice=%s, language=%s)",
        agent_display_name, voice_id or DEFAULT_TTS_MODEL, language,
    )

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Parsear "provider/model:voice-id" para inyectar language directamente en InferenceTTS
    model = voice_id if voice_id else DEFAULT_TTS_MODEL
    voice: str | None = None
    if voice_id and ":" in voice_id:
        idx = voice_id.rfind(":")
        voice = voice_id[idx + 1:]
        model = voice_id[:idx]

    tts_kwargs: dict = {"model": model}
    if voice:
        tts_kwargs["voice"] = voice
    if not model.startswith("deepgram/"):
        tts_kwargs["language"] = language

    tts = InferenceTTS(**tts_kwargs)

    logger.info("Waiting for SIP participant to join the room…")
    sip_participant = await _wait_for_sip_participant(ctx)

    if sip_participant is None:
        logger.error("SIP participant never joined — ending session")
        return

    logger.info("SIP participant joined: %s — starting agent session", sip_participant.identity)

    if language == "es":
        stt = InferenceSTT(model="deepgram/nova-2", language="es")
    else:
        stt = InferenceSTT(model="deepgram/nova-2-phonecall")

    llm_api_key = (
        os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    ).strip()

    if not llm_api_key:
        logger.error("CRITICAL: No OPENROUTER_API_KEY or OPENAI_API_KEY found in .env!")

    agent = Agent(
        instructions=agent_instructions,
        stt=stt,
        llm=openai.LLM(
            model="anthropic/claude-haiku-5.5",
            base_url="https://openrouter.ai/api/v1",
            api_key=llm_api_key,
            temperature=0.6,
        ),
        tts=tts,
        vad=silero.VAD.load(
            min_silence_duration=0.35,
            activation_threshold=0.60,
        ),
    )

    session = AgentSession(
        min_endpointing_delay=0.25,
        max_endpointing_delay=1.2,
    )

    last_metrics: dict = {"eou_ms": 0.0, "llm_ttft_ms": 0.0, "tts_ttfb_ms": 0.0}

    @session.on("metrics_collected")
    def on_metrics_collected(event):
        m = getattr(event, "metrics", event)

        if hasattr(m, "end_of_utterance_delay"):
            eou_ms = getattr(m, "end_of_utterance_delay", 0.0) * 1000
            stt_ms = getattr(m, "transcription_delay", 0.0) * 1000
            last_metrics["eou_ms"] = eou_ms
            print(
                f"⏱️  [TELEMETRÍA EOU/STT] Fin de turno (EOU): {eou_ms:.0f}ms | Transcripción STT: {stt_ms:.0f}ms",
                flush=True,
            )
        elif hasattr(m, "ttft"):
            ttft_ms = getattr(m, "ttft", 0.0) * 1000
            tps = getattr(m, "tokens_per_second", 0.0)
            last_metrics["llm_ttft_ms"] = ttft_ms
            print(
                f"🧠 [TELEMETRÍA LLM] TTFT (1er Token): {ttft_ms:.0f}ms | Velocidad: {tps:.1f} tok/s",
                flush=True,
            )
        elif hasattr(m, "ttfb"):
            ttfb_ms = getattr(m, "ttfb", 0.0) * 1000
            last_metrics["tts_ttfb_ms"] = ttfb_ms
            total_v2v = (
                last_metrics["eou_ms"]
                + last_metrics["llm_ttft_ms"]
                + last_metrics["tts_ttfb_ms"]
            )
            print(
                f"🔊 [TELEMETRÍA TTS] TTFB (1er Audio): {ttfb_ms:.0f}ms | ⚡ LATENCIA V2V TOTAL: ~{total_v2v:.0f}ms",
                flush=True,
            )

    @session.on("conversation_item_added")
    def on_conversation_item_added(event):
        item = getattr(event, "item", event)
        role = getattr(item, "role", None)
        text = getattr(item, "text_content", None) or getattr(item, "content", "")
        if isinstance(text, list):
            text = " ".join(str(part) for part in text if part)
        if role == "user":
            print(f"\n👤 [{contact_name or 'Usuario'}]: {text}", flush=True)
        elif role == "assistant":
            print(f"\n🤖 [{agent_display_name}]: {text}\n", flush=True)

    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            participant_identity=sip_participant.identity,
        ),
    )

    # Handshake telefónico en 2 fases: primero confirmar línea antes de lanzar el pitch
    if language == "es":
        if contact_name:
            greeting = (
                f"Acaban de contestar el teléfono. Di únicamente esta frase exacta, sin signos de exclamación: "
                f"'Aló... hola, ¿hablo con {contact_name}?' "
                "No digas tu nombre ni el motivo de la llamada todavía. Espera a que te responda."
            )
        else:
            greeting = (
                "Acaban de contestar el teléfono. Di únicamente: 'Aló, buenas... ¿me escuchas bien?' "
                "No digas nada más hasta que la persona responda."
            )
    else:
        if contact_name:
            greeting = f"Say only: 'Hello... is this {contact_name}?' Wait for their response before introducing yourself."
        else:
            greeting = "Say only: 'Hello, hi there... can you hear me okay?' Wait for their response."

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