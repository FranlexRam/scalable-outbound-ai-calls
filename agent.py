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


async def _hangup_room(room_name: str) -> None:
    """Destruye la sala en LiveKit Cloud para forzar el corte SIP BYE inmediato en Twilio."""
    if not room_name or room_name in ("console", "console-room", "mock_room"):
        return
    lk_url = os.environ.get("LIVEKIT_URL", "")
    lk_key = os.environ.get("LIVEKIT_API_KEY", "")
    lk_secret = os.environ.get("LIVEKIT_API_SECRET", "")
    if not (lk_url and lk_key and lk_secret):
        return
    lk = api.LiveKitAPI(url=lk_url, api_key=lk_key, api_secret=lk_secret)
    try:
        await lk.room.delete_room(api.DeleteRoomRequest(room=room_name))
        logger.info("🔒 [KILL-SWITCH] Room '%s' deleted in LiveKit Cloud (SIP call terminated).", room_name)
    except Exception as exc:
        logger.debug("Room cleanup note for %s: %s", room_name, exc)
    finally:
        await lk.aclose()


async def _terminate_call_session(
    ctx: JobContext,
    session: AgentSession,
    delay: float = 5.5,
) -> None:
    """Espera a que termine de reproducirse toda la frase de despedida, destruye la sala SIP/Cloud y apaga la sesión."""
    try:
        session.input.set_audio_enabled(False)
    except Exception:
        pass

    # Esperar primero a que el LLM y TTS generen y reproduzcan el audio de despedida
    await asyncio.sleep(1.5)
    try:
        if getattr(session, "current_speech", None) is not None:
            await asyncio.wait_for(session.current_speech.wait_for_playout(), timeout=delay)
        else:
            await asyncio.sleep(delay - 1.5)
    except Exception:
        await asyncio.sleep(3.5)

    await asyncio.sleep(0.8)
    await _hangup_room(ctx.room.name)

    try:
        await session.aclose()
    except Exception:
        pass

    ctx.shutdown(reason="Call terminated cleanly by end_call")


async def _call_duration_watchdog(room_name: str, max_seconds: int) -> None:
    """Corta la llamada automáticamente si supera el tiempo máximo permitido."""
    try:
        await asyncio.sleep(max_seconds)
        logger.warning(
            "⏰ [WATCHDOG] Max call duration (%ds) reached for room '%s' — forcing hangup.",
            max_seconds,
            room_name,
        )
        await _hangup_room(room_name)
    except asyncio.CancelledError:
        pass


async def entrypoint(ctx: JobContext):
    load_dotenv(override=True)

    metadata: dict = {}
    if ctx.job.metadata:
        try:
            metadata = json.loads(ctx.job.metadata)
        except json.JSONDecodeError:
            logger.warning("Could not parse job metadata as JSON")

    # Valores por defecto amigables para pruebas locales en modo consola o web
    phone_number: str = metadata.get("phone_number", "")
    agent_instructions: str = metadata.get(
        "agent_instructions",
        (
            "Eres asesora de atención al cliente de Zen Spa. Cuando el cliente confirme que es él después del aló, "
            "salúdalo con calma, dile que lo llamas rapidito de Zen Spa para regalarle un cincuenta por ciento de descuento "
            "en depilación láser por ser cliente de la casa, y pregúntale si tiene un minutito. "
            "Si acepta agendar la valoración gratuita, pregúntale qué día de la semana le queda más cómodo, en la mañana o en la tarde."
        ),
    )
    agent_display_name: str = metadata.get("agent_name", "Daniela")
    voice_id: str | None = metadata.get("voice_id") or "cartesia/sonic-2:5c5ad5e7-1020-476b-8b91-fdcbe9cc313c"
    language: str = metadata.get("language") or "es"
    contact_name: str = metadata.get("contact_name", "Franlex")
    personalization_context: str = metadata.get(
        "personalization_context",
        "Cliente frecuente de Zen Spa. Beneficio de cincuenta por ciento de descuento en paquetes de depilación láser definitiva.",
    )

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
            "6. FLUJO DESPUÉS DEL ALÓ Y CIERRE: Cuando el cliente responda a tu '¿Aló?', preséntate brevemente, menciona el motivo en una frase sencilla y pregúntale si tiene un minutito. Cuando la conversación termine y te despidas (o si te dicen número equivocado / no interesado), di tu frase de despedida e invoca la herramienta end_call en ese mismo turno."
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
            "6. POST-HELLO FLOW AND HANGUP: Once the person replies to your initial 'Hello?', introduce yourself briefly, mention the main reason for calling in one simple phrase, and ask if caught them at a good time. When the conversation concludes and you say goodbye, invoke the end_call tool in that same turn."
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

    # Detectar si estamos en modo consola local (gratis) o en sala LiveKit (SIP / WebRTC)
    is_console_mode = (
        ctx.room.name in ("console", "console-room", "mock_room", "")
        or (len(sys_argv := os.sys.argv) > 1 and "console" in sys_argv)
    )
    target_participant = None

    if not is_console_mode:
        logger.info("Waiting for participant (SIP or WebRTC) to join the room…")
        target_participant = await _wait_for_call_participant(ctx)

        if target_participant is None:
            logger.error("Participant never joined — destroying room to prevent zombie billing")
            await _hangup_room(ctx.room.name)
            return

        logger.info("Participant joined: %s — starting agent session", target_participant.identity)

    if language == "es":
        stt = InferenceSTT(model="deepgram/nova-3", language="es")
    else:
        stt = InferenceSTT(model="deepgram/nova-3", language="en")

    llm_api_key = (
        os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    ).strip()

    if not llm_api_key:
        logger.error("CRITICAL: No OPENROUTER_API_KEY or OPENAI_API_KEY found in .env!")

    session = AgentSession(
        min_endpointing_delay=0.20,
        max_endpointing_delay=0.60,
    )

    call_ending = False

    @llm.function_tool(
        description=(
            "End and hang up the call. Invoke this tool in the exact same turn where you say your final goodbye "
            "(when an appointment is confirmed and you say goodbye, or when the user is not interested / wrong number)."
        )
    )
    async def end_call() -> None:
        nonlocal call_ending
        if call_ending:
            return None
        call_ending = True
        logger.info("📴 Agent triggered end_call tool — waiting for goodbye audio to finish before hanging up…")
        asyncio.create_task(_terminate_call_session(ctx, session, delay=5.5))
        return None

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
            min_silence_duration=0.30,
            activation_threshold=0.60,
        ),
        tools=[end_call],
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

    room_input_opts = (
        RoomInputOptions(participant_identity=target_participant.identity)
        if target_participant is not None
        else RoomInputOptions()
    )

    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=room_input_opts,
    )

    # Watchdog de seguridad financiera: corta la llamada si supera max_call_duration (por defecto 180s)
    max_duration = int(metadata.get("max_call_duration", os.environ.get("MAX_CALL_DURATION_SECONDS", "180")))
    watchdog_task = asyncio.create_task(_call_duration_watchdog(ctx.room.name, max_duration))

    async def _on_shutdown_cleanup() -> None:
        watchdog_task.cancel()
        await _hangup_room(ctx.room.name)

    ctx.add_shutdown_callback(_on_shutdown_cleanup)

    # Si el participante cuelga desde su teléfono o navegador, destruir la sala inmediatamente
    @ctx.room.on("participant_disconnected")
    def on_participant_disconnected(participant):
        if target_participant and participant.identity == target_participant.identity:
            logger.info("📴 Participant %s disconnected — destroying room immediately.", participant.identity)
            asyncio.create_task(_terminate_call_session(ctx, session, delay=0.1))

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


async def _wait_for_call_participant(ctx: JobContext, timeout: float = 60.0):
    """Acepta tanto participantes telefónicos (sip_) como de prueba web gratuita (web_)."""
    for p in ctx.room.remote_participants.values():
        if p.identity.startswith(("sip_", "web_", "user")):
            return p

    future: asyncio.Future = asyncio.get_event_loop().create_future()

    def on_participant_connected(participant):
        if participant.identity.startswith(("sip_", "web_", "user")) and not future.done():
            future.set_result(participant)

    ctx.room.on("participant_connected", on_participant_connected)

    try:
        return await asyncio.wait_for(future, timeout=timeout)
    except asyncio.TimeoutError:
        logger.error("Timed out waiting for participant to join")
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