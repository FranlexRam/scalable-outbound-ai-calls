import asyncio
import calendar
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
from livekit.agents.inference import (
    LLM as InferenceLLM,
    STT as InferenceSTT,
    TTS as InferenceTTS,
)
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
DEFAULT_LIVEKIT_LLM_MODEL = os.environ.get("LIVEKIT_LLM_MODEL", "openai/gpt-4o-mini")
DEFAULT_OPENROUTER_LLM_MODEL = os.environ.get("OPENROUTER_LLM_MODEL", "anthropic/claude-haiku-5.5")

DAYS_ES = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
MONTHS_ES = [
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
]


def _build_three_month_calendar(now: datetime, language: str) -> str:
    """Genera dinámicamente desde now() el calendario exacto del mes actual y los próximos 2 meses para conciencia temporal absoluta."""
    cal = calendar.TextCalendar(calendar.MONDAY)
    blocks: list[str] = []
    year, month = now.year, now.month
    for _ in range(3):
        if language == "es":
            month_title = f"{MONTHS_ES[month - 1].capitalize()} {year}"
            lines = cal.formatmonth(year, month).strip().splitlines()
            if len(lines) >= 2:
                lines[0] = month_title
                lines[1] = "Lu Ma Mi Ju Vi Sa Do"
            blocks.append("\n".join(lines))
        else:
            blocks.append(cal.formatmonth(year, month).strip())
        month += 1
        if month > 12:
            month = 1
            year += 1
    return "\n\n".join(blocks)


def _get_temporal_context(language: str) -> str:
    """Calcula la fecha, día y hora actual exacta (America/Caracas) y el calendario completo desde now() sin límite de días."""
    now = datetime.now(ZoneInfo("America/Caracas"))
    cal_str = _build_three_month_calendar(now, language)
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
            f"CONCIENCIA TEMPORAL INTERNA ABSOLUTA (PROHIBIDO mencionar zonas horarias o países en voz alta):\n"
            f"- Fecha y hora actual (now): Hoy es {day_name} {now.day} de {month_name} de {now.year} "
            f"(ISO: {now.strftime('%Y-%m-%d')}), y son las {hour_12}:{now.minute:02d} {period}.\n"
            f"- Referencia exacta de calendario oficial desde hoy (útil para saber qué fecha cae cualquier día de esta semana, la próxima, o dentro de varias semanas/meses sin equivocarte jamás):\n"
            f"{cal_str}"
        )
    return (
        f"ABSOLUTE INTERNAL TIME AWARENESS (NEVER mention timezones or countries out loud):\n"
        f"- Current date and time (now): Today is {now.strftime('%A, %B %d, %Y')} "
        f"(ISO: {now.strftime('%Y-%m-%d')}), and the time is {now.strftime('%I:%M %p')}.\n"
        f"- Official calendar reference starting from today (use this to know the exact date of any day this week, next week, or several weeks/months ahead with 100% accuracy):\n"
        f"{cal_str}"
    )


def _build_llm() -> llm.LLM:
    """Construye el motor LLM de Producción: Primary = LiveKit Inference (Virginia ~400ms) | Fallback = Claude Haiku 5.5 (OpenRouter Latency-Sorted)."""
    llm_api_key = (
        os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    ).strip()

    openrouter_llm: llm.LLM | None = None
    if llm_api_key:
        try:
            openrouter_llm = openai.LLM(
                model=DEFAULT_OPENROUTER_LLM_MODEL,
                base_url="https://openrouter.ai/api/v1",
                api_key=llm_api_key,
                temperature=0.5,
                extra_body={
                    "provider": {
                        "sort": "latency",
                        "allow_fallbacks": True,
                    },
                },
            )
        except TypeError:
            openrouter_llm = openai.LLM(
                model=DEFAULT_OPENROUTER_LLM_MODEL,
                base_url="https://openrouter.ai/api/v1",
                api_key=llm_api_key,
                temperature=0.5,
            )

    livekit_llm: llm.LLM | None = None
    if os.environ.get("USE_LIVEKIT_LLM", "1") != "0":
        try:
            livekit_llm = InferenceLLM(
                model=DEFAULT_LIVEKIT_LLM_MODEL,
                extra_kwargs={"temperature": 0.5},
            )
        except TypeError:
            try:
                livekit_llm = InferenceLLM(model=DEFAULT_LIVEKIT_LLM_MODEL)
            except Exception as exc:
                logger.warning("Could not initialize LiveKit InferenceLLM: %s", exc)
        except Exception as exc:
            logger.warning("Could not initialize LiveKit InferenceLLM: %s", exc)

    if livekit_llm is not None and openrouter_llm is not None:
        if hasattr(llm, "FallbackAdapter"):
            for kwarg_name in ("llm", "llms"):
                try:
                    adapter = llm.FallbackAdapter(**{kwarg_name: [livekit_llm, openrouter_llm]})
                    logger.info(
                        "🧠 Production Dual LLM active: Primary='%s' (LiveKit Virginia ~400ms) | Fallback='%s' (OpenRouter Latency-Sorted)",
                        DEFAULT_LIVEKIT_LLM_MODEL,
                        DEFAULT_OPENROUTER_LLM_MODEL,
                    )
                    return adapter
                except TypeError:
                    continue
                except Exception as exc:
                    logger.debug("FallbackAdapter note: %s", exc)
                    break
        logger.info("🧠 Primary LLM Engine active: '%s' (LiveKit Virginia)", DEFAULT_LIVEKIT_LLM_MODEL)
        return livekit_llm

    if livekit_llm is not None:
        logger.info("🧠 Primary LLM Engine active: '%s' (LiveKit Virginia)", DEFAULT_LIVEKIT_LLM_MODEL)
        return livekit_llm

    if openrouter_llm is not None:
        logger.info("🧠 Fallback LLM Engine active: '%s' (OpenRouter Lowest-Latency)", DEFAULT_OPENROUTER_LLM_MODEL)
        return openrouter_llm

    raise RuntimeError("CRITICAL: Neither OpenRouter LLM nor LiveKit InferenceLLM could be initialized!")


def prewarm(proc: JobProcess) -> None:
    """Precalienta recursos pesados (Silero VAD y toolsets) en RAM antes de que entre la llamada."""
    proc.userdata["vad"] = silero.VAD.load(
        min_silence_duration=0.30,
        activation_threshold=0.65,
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
    max_wait: float = 6.5,
) -> None:
    """Monitorea en tiempo real el fin de la locución de despedida y corta la llamada sin silencios muertos."""
    try:
        session.input.set_audio_enabled(False)
    except Exception:
        pass

    # Dar margen breve (0.9s) para que el TTS inicie el playout de la frase de despedida
    await asyncio.sleep(0.9)

    loop = asyncio.get_event_loop()
    deadline = loop.time() + max_wait
    while loop.time() < deadline:
        speech = getattr(session, "current_speech", None)
        if speech is None:
            break
        if getattr(speech, "done", lambda: False)():
            break
        try:
            await asyncio.wait_for(speech.wait_for_playout(), timeout=0.4)
            break
        except asyncio.TimeoutError:
            continue
        except Exception:
            break

    # Pausa natural breve tras la última sílaba antes de colgar
    await asyncio.sleep(0.6)
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

    # Valores por defecto para pruebas locales en consola (cuando se llama desde la UI web, metadata los reemplaza dinámicamente)
    phone_number: str = metadata.get("phone_number", "")
    agent_instructions: str = metadata.get(
        "agent_instructions",
        (
            "Representas a Zen Spa. El objetivo es invitar al cliente a aprovechar su cincuenta por ciento de descuento "
            "en paquetes de depilación láser definitiva y agendarle una sesión de valoración diagnóstica gratuita de quince minutos "
            "con nuestra especialista para evaluar su piel sin compromiso."
        ),
    )
    agent_display_name: str = metadata.get("agent_name", "Daniela")
    voice_id: str | None = metadata.get("voice_id") or "cartesia/sonic-2:5c5ad5e7-1020-476b-8b91-fdcbe9cc313c"
    language: str = metadata.get("language") or "es"
    contact_name: str = metadata.get("contact_name", "Franlex")
    personalization_context: str = metadata.get(
        "personalization_context",
        "Cliente frecuente y leal de Zen Spa. Tiene pre-aprobado un cincuenta por ciento de descuento en depilación láser definitiva y valoración inicial sin costo.",
    )

    temporal_context = _get_temporal_context(language)
    raw_business_instructions = agent_instructions.strip()
    raw_call_context = personalization_context.strip()
    client_label = contact_name.strip() if contact_name else ("Cliente" if language == "es" else "Prospect")

    if language == "es":
        agent_instructions = (
            "<runtime_data>\n"
            f"  <temporal_awareness>\n{temporal_context}\n  </temporal_awareness>\n"
            "  <business_and_client_context>\n"
            f"    <client_name>{client_label}</client_name>\n"
            f"    <call_context>{raw_call_context}</call_context>\n"
            f"    <business_instructions>{raw_business_instructions}</business_instructions>\n"
            "  </business_and_client_context>\n"
            "</runtime_data>\n\n"
            "<agent_system_manifest>\n"
            "  <identity>\n"
            f"    Eres {agent_display_name}, Senior Client Success Manager y especialista en ventas consultivas telefónicas. "
            "Tu tono es cálido, empático, natural y ejecutivo. Tratas al cliente de tú con cercanía sin repetir su nombre en todos los turnos.\n"
            "  </identity>\n\n"
            "  <execution_pipeline>\n"
            "    <phase index=\"1\" name=\"apertura_con_motivo_y_beneficio\">\n"
            "      En cuanto el cliente confirme que es él después del 'Aló', NUNCA preguntes '¿tienes un minuto?' a secas sin dar contexto. "
            "En 2 oraciones fluidas (máximo 28 palabras): saluda, menciona de dónde llamas, explica POR QUÉ lo llamas y cuál es su beneficio especial según <call_context> y <business_instructions>, y cierra preguntando suavemente si tiene medio minutito.\n"
            "    </phase>\n"
            "    <phase index=\"2\" name=\"explicacion_de_valor_antes_de_horarios\">\n"
            "      Cuando el cliente diga 'Sí, ok' o pregunte '¿De qué trata? / ¿Por qué me llamas?', NUNCA lances horarios de golpe en frío. "
            "Primero explica en 1 oración clara (máximo 25 palabras) en qué consiste el beneficio o paso propuesto en <business_instructions> para que entienda el valor sin compromiso, y cierra preguntando qué día o semana le quedaría más cómodo.\n"
            "    </phase>\n"
            "    <phase index=\"3\" name=\"negociacion_consultiva_de_agenda\">\n"
            "      - Cuando el cliente indique qué día o semana prefiere, verifica la fecha exacta en <temporal_awareness> y guíalo proponiendo dos horas concretas (variando horas entre mañana y tarde).\n"
            "      - Si el cliente propone otra hora específica razonable (ej. 'mejor a las diez' o 'a las once'), adáptate de inmediato como un asesor senior y acéptala sin negársela.\n"
            "      - Si dice que no puede en los horarios sugeridos, JAMÁS des por perdida la llamada ni te despidas: empatiza ('Claro, te entiendo perfecto.') y pregúntale '¿Qué hora te queda mejor a ti?'.\n"
            "    </phase>\n"
            "    <phase index=\"4\" name=\"confirmacion_y_cierre\">\n"
            "      ÚNICAMENTE cuando el día, fecha y hora exacta estén confirmados por el cliente, o cuando rechace explícitamente el servicio o la llamada ('no me interesa el servicio', 'estoy muy ocupado, adiós', 'número equivocado'), confirma el acuerdo en una frase breve, despídete amablemente y ejecuta la herramienta de colgar llamada en ese mismo turno.\n"
            "    </phase>\n"
            "  </execution_pipeline>\n\n"
            "  <hard_constraints>\n"
            "    - MICRO-APERTURA TTS: Inicia siempre tu respuesta con una micro-frase de 2 a 3 palabras cerrada con PUNTO SEGUIDO ('Claro, buen día.', 'Perfecto, mira.', 'Ah, te cuento.') antes de la idea principal para que tu voz suene al instante.\n"
            "    - LONGITUD ADAPTATIVA: Usa hasta 28 palabras en las Fases 1 y 2 para que el cliente comprenda el motivo y valor; usa de 12 a 18 palabras en las Fases 3 y 4 para agendar con agilidad.\n"
            "    - FONÉTICA ESTRICTA: Responde en un solo párrafo continuo sin saltos de línea. PROHIBIDO usar signos de exclamación (¡ !) o puntos suspensivos al inicio. Escribe TODOS los números, fechas, porcentajes y horas en letras minúsculas ('cincuenta por ciento', 'diecinueve de noviembre', 'diez de la mañana').\n"
            "    - AISLAMIENTO DE HERRAMIENTAS: JAMÁS escribas ni pronuncies nombres de funciones o código en tu respuesta de texto.\n"
            "  </hard_constraints>\n"
            "</agent_system_manifest>"
        )
    else:
        agent_instructions = (
            "<runtime_data>\n"
            f"  <temporal_awareness>\n{temporal_context}\n  </temporal_awareness>\n"
            "  <business_and_client_context>\n"
            f"    <client_name>{client_label}</client_name>\n"
            f"    <call_context>{raw_call_context}</call_context>\n"
            f"    <business_instructions>{raw_business_instructions}</business_instructions>\n"
            "  </business_and_client_context>\n"
            "</runtime_data>\n\n"
            "<agent_system_manifest>\n"
            "  <identity>\n"
            f"    You are {agent_display_name}, a Senior Client Success Manager and consultative outbound sales specialist. "
            "Your tone is warm, empathetic, natural, and executive. Do not overuse the prospect's name in every turn.\n"
            "  </identity>\n\n"
            "  <execution_pipeline>\n"
            "    <phase index=\"1\" name=\"contextual_opening_with_reason_and_benefit\">\n"
            "      As soon as the prospect confirms it is them after 'Hello?', NEVER ask 'do you have a minute?' without context. "
            "In 2 smooth sentences (max 28 words): greet them, state where you are calling from, explain WHY you are calling and their specific benefit based on <call_context> and <business_instructions>, and softly ask if they have half a minute.\n"
            "    </phase>\n"
            "    <phase index=\"2\" name=\"value_explanation_before_scheduling\">\n"
            "      When the prospect says 'Sure / Okay' or asks 'What is this about? / Why is it free?', NEVER push time slots cold. "
            "First explain in 1 clear sentence (max 25 words) how the benefit or next step in <business_instructions> works so they understand the value with no obligation, then ask what day or week works best for them.\n"
            "    </phase>\n"
            "    <phase index=\"3\" name=\"consultative_schedule_negotiation\">\n"
            "      - Once the prospect mentions a preferred day or week, verify the exact date in <temporal_awareness> and guide them by offering two concrete times.\n"
            "      - If the prospect suggests another reasonable time (e.g., 'ten am works better'), adapt immediately like a senior advisor and accept their slot.\n"
            "      - If neither suggested time works, NEVER give up or end the call: empathize ('Got it, totally understand.') and ask 'What time works best for you?'.\n"
            "    </phase>\n"
            "    <phase index=\"4\" name=\"confirmation_and_hangup\">\n"
            "      ONLY when the exact day, date, and time are confirmed, or when the prospect explicitly rejects the service or call ('not interested in the service', 'too busy, goodbye', 'wrong number'), confirm the details in one warm sentence, say goodbye, and execute the hangup tool in that exact same turn.\n"
            "    </phase>\n"
            "  </execution_pipeline>\n\n"
            "  <hard_constraints>\n"
            "    - TTS MICRO-OPENER: Always start your turn with a 2-to-3 word micro-phrase ending in a PERIOD ('Sure, good morning.', 'Perfect, so look.', 'Oh, got it.') before your main sentence so your voice starts instantly.\n"
            "    - ADAPTIVE LENGTH: Use up to 28 words in Phases 1 and 2 so the prospect clearly understands the reason and value; use 12 to 18 words in Phases 3 and 4 for fast scheduling.\n"
            "    - STRICT PHONETICS: Reply in a single continuous paragraph without line breaks. NEVER use exclamation marks (!) or leading ellipses (...). Spell out ALL numbers, dates, percentages, and times in lowercase words ('fifty percent', 'november nineteenth', 'ten in the morning').\n"
            "    - TOOL ISOLATION: NEVER write or speak function names or code in your text response.\n"
            "  </hard_constraints>\n"
            "</agent_system_manifest>"
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

    llm_engine = _build_llm()

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
            "Hang up and terminate the phone call. Call this tool immediately in the same turn when the appointment date and time are confirmed and you say goodbye, or when the user explicitly rejects the service."
        )
    )
    async def end_call() -> None:
        nonlocal call_ending
        if call_ending:
            return None
        call_ending = True
        logger.info("📴 Agent triggered end_call tool — waiting for goodbye audio to finish before hanging up…")
        asyncio.create_task(_terminate_call_session(ctx, session, max_wait=6.5))
        return None

    vad_instance = ctx.proc.userdata.get("vad") or silero.VAD.load(
        min_silence_duration=0.30,
        activation_threshold=0.65,
    )

    agent = Agent(
        instructions=agent_instructions,
        stt=stt,
        llm=llm_engine,
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
        nonlocal call_ending
        item = getattr(event, "item", event)
        role = getattr(item, "role", None)
        text = getattr(item, "text_content", None) or getattr(item, "content", "")
        if isinstance(text, list):
            text = " ".join(str(part) for part in text if part)
        if role == "user":
            print(f"\n👤 [{contact_name or 'Usuario'}]: {text}", flush=True)
        elif role == "assistant":
            clean_text = text.replace("functions.end_call", "").replace("end_call()", "").strip()
            print(f"\n🤖 [{agent_display_name}]: {clean_text}\n", flush=True)
            if not call_ending and ("functions.end_call" in text or "end_call(" in text):
                call_ending = True
                logger.info("📴 Safety-net triggered end_call from assistant text — waiting for goodbye audio…")
                asyncio.create_task(_terminate_call_session(ctx, session, max_wait=6.5))

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
            asyncio.create_task(_terminate_call_session(ctx, session, max_wait=0.2))

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