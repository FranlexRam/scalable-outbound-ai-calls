import asyncio
from datetime import datetime
import json
import logging
import os
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from livekit import api
from livekit.protocol.sip import SIP_MEDIA_ENCRYPT_DISABLE
from livekit.agents import (
    AgentSession,
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.inference import TTS as InferenceTTS, STT as InferenceSTT
from livekit.agents.voice import Agent
from livekit.agents.voice.room_io import RoomOptions
from livekit.plugins import openai, silero

# Pre-importar async_toolset para evitar el bloqueo de 159ms en el Event Loop al iniciar sesión
try:
    import livekit.agents.llm.async_toolset  # noqa: F401
except ImportError:
    pass

try:
    from livekit.agents.voice.agent_session import TurnHandlingOptions
except ImportError:
    TurnHandlingOptions = None

load_dotenv(override=True)

logger = logging.getLogger("outbound-agent")
logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("livekit").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

AGENT_NAME = "outbound-agent"
SIP_TRUNK_ID = os.environ.get("LIVEKIT_SIP_TRUNK_ID", "")
DEFAULT_TTS_MODEL = "cartesia/sonic-2"

DAYS_ES = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
MONTHS_ES = [
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
]


def _get_temporal_context(language: str) -> str:
    """Calcula la fecha, día y hora actual exacta (America/Caracas) como contexto interno silencioso."""
    now = datetime.now(ZoneInfo("America/Caracas"))
    if language == "es":
        day_name = DAYS_ES[now.weekday()]
        month_name = MONTHS_ES[now.month - 1]
        period = (
            "de la mañana"
            if now.hour < 12
            else ("de la tarde" if now.hour < 19 else "de la noche")
        )
        hour_12 = now.hour % 12 or 12
        return (
            f"CONCIENCIA TEMPORAL INTERNA (PROHIBIDO mencionar zonas horarias o países): "
            f"Ahora mismo es {day_name} {now.day} de {month_name} de {now.year}, {hour_12}:{now.minute:02d} {period}. "
            "Úsalo silenciosamente para saludar según la hora actual y calcular fechas reales al agendar ('hoy', 'mañana', 'este viernes')."
        )
    return (
        f"INTERNAL TIME AWARENESS (NEVER mention timezones or countries out loud): "
        f"Right now it is {now.strftime('%A, %B %d, %Y, %I:%M %p')}. "
        "Use this silently to greet appropriately and resolve relative dates ('today', 'tomorrow', 'this Friday') when booking."
    )


def prewarm(proc: JobProcess) -> None:
    """Precalienta recursos pesados (Silero VAD y toolsets) en RAM antes de que entre la llamada."""
    proc.userdata["vad"] = silero.VAD.load(
        min_silence_duration=0.30,
        activation_threshold=0.60,
    )


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
            "Eres asesora de Zen Spa. Cuando el cliente confirme que es él después del aló, "
            "dile que lo llamas rapidito de Zen Spa para regalarle un cincuenta por ciento de descuento "
            "en depilación láser con valoración gratuita esta semana, y pregúntale si tiene un minutito."
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

    temporal_context = _get_temporal_context(language)

    if language == "es":
        agent_instructions = (
            f"Tu identidad: Eres {agent_display_name}, especialista senior en ventas y agendamiento telefónico.\n"
            f"{temporal_context}\n\n"
            f"OBJETIVO DEL NEGOCIO:\n{agent_instructions}"
        )

        if contact_name:
            agent_instructions += f"\nCliente en línea: {contact_name} (trátalo de tú con calidez, sin repetir su nombre en cada turno)."

        if personalization_context:
            agent_instructions += f"\nContexto del cliente: {personalization_context}"

        agent_instructions += (
            "\n\nREGLAS DE ORO DE VENTAS TELEFÓNICAS Y VOZ (OBLIGATORIO):\n"
            "1. CLIENTE OCUPADO (MÁXIMO 16 PALABRAS): Quien contesta está ocupado. Responde SIEMPRE en 1 sola oración corta y directa (12 a 18 palabras máximo) cerrando con una pregunta breve.\n"
            "2. CIERRE DE AGENDA EN VIVO: NUNCA digas 'luego reviso la agenda y te escribo' ni 'te confirmo la hora por mensaje'. Tú tienes disponibilidad en vivo: cuando el cliente elija un día o turno (mañana/tarde), propón inmediatamente dos horas concretas (ej: '¿Te queda mejor a las diez o a las once de la mañana?').\n"
            "3. MANEJO DE OBJECIONES RÁPIDO: Si pregunta de qué trata o si duele/cuesta, responde el beneficio clave en menos de 10 palabras y retoma el cierre con doble opción de horario.\n"
            "4. NATURALIDAD Y FONÉTICA TTS: Español neutro latinoamericano, cálido y relajado. PROHIBIDO usar signos de exclamación (¡ !). Escribe TODOS los números, porcentajes y horas en letras minúsculas ('cincuenta por ciento', 'cuatro de la tarde'). Inicia con conectores cortos seguidos de coma ('Claro, mira,', 'Ah, entiendo,', 'Perfecto,') sin puntos suspensivos al inicio.\n"
            "5. CIERRE Y CORTE (end_call): En cuanto el cliente confirme la hora exacta (o si dice que no le interesa / está muy ocupado / número equivocado), confirma día y hora en una frase corta, despídete amablemente e invoca la herramienta end_call en ese mismo turno."
        )
    else:
        agent_instructions = (
            f"Your identity: You are {agent_display_name}, a senior outbound sales and appointment specialist.\n"
            f"{temporal_context}\n\n"
            f"BUSINESS OBJECTIVE:\n{agent_instructions}"
        )

        if contact_name:
            agent_instructions += f"\nProspect on the line: {contact_name} (be warm and natural, do not overuse their name)."

        if personalization_context:
            agent_instructions += f"\nProspect context: {personalization_context}"

        agent_instructions += (
            "\n\nMANDATORY OUTBOUND SALES & VOICE RULES:\n"
            "1. BUSY PROSPECT (MAX 16 WORDS): The person is busy. ALWAYS reply in 1 ultra-concise sentence (12 to 18 words max) ending with a quick question.\n"
            "2. LIVE CALENDAR CLOSING: NEVER say 'I will check the schedule and text you later'. You control the schedule live: as soon as they pick a day or part of day, immediately offer two concrete times (e.g., 'Does ten in the morning or two in the afternoon work better?').\n"
            "3. FAST OBJECTION HANDLING: Answer any question in under 10 words focusing on value, then immediately pivot back to a two-option time close.\n"
            "4. TTS PHONETICS & TONE: Warm, relaxed office tone. NEVER use exclamation marks (!). Spell out ALL numbers, percentages, and times in lowercase words ('fifty percent', 'four in the afternoon'). Start with short fillers followed by a comma ('Sure, so,', 'Oh, got it,') with no ellipses at the start.\n"
            "5. HANGUP (end_call): Once the exact day and time are confirmed (or if not interested / busy / wrong number), confirm the slot in one brief phrase, say a warm goodbye, and invoke the end_call tool in that exact same turn."
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

    session: AgentSession
    if TurnHandlingOptions is not None:
        try:
            session = AgentSession(
                turn_handling=TurnHandlingOptions(
                    endpointing={
                        "mode": "dynamic",
                        "min_delay": 0.25,
                        "max_delay": 0.60,
                    },
                    preemptive_generation=True,
                ),
            )
        except Exception:
            session = AgentSession(
                min_endpointing_delay=0.25,
                max_endpointing_delay=0.60,
                preemptive_generation=True,
            )
    else:
        session = AgentSession(
            min_endpointing_delay=0.25,
            max_endpointing_delay=0.60,
            preemptive_generation=True,
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

    vad_instance = ctx.proc.userdata.get("vad") or silero.VAD.load(
        min_silence_duration=0.30,
        activation_threshold=0.60,
    )

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
        vad=vad_instance,
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

    room_opts = (
        RoomOptions(participant_identity=target_participant.identity)
        if target_participant is not None
        else RoomOptions()
    )

    await session.start(
        agent=agent,
        room=ctx.room,
        room_options=room_opts,
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
            prewarm_fnc=prewarm,
            agent_name=AGENT_NAME,
        )
    )