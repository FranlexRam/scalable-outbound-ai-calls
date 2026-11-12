import asyncio
from datetime import datetime, timedelta
import json
import logging
import os
import re
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
SECONDARY_OPENROUTER_LLM_MODEL = os.environ.get("SECONDARY_OPENROUTER_LLM_MODEL", "google/gemini-2.5-flash")
QUOTA_STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".lk_quota_state.json")


def _current_year_month() -> str:
    return datetime.now(ZoneInfo("America/Caracas")).strftime("%Y-%m")


def _is_livekit_quota_exhausted_this_month() -> bool:
    """Devuelve True si los créditos gratis del proyecto actual de LiveKit se agotaron este mes; se resetea al cambiar de mes o de proyecto."""
    current_url = os.environ.get("LIVEKIT_URL", "").strip()
    try:
        if os.path.exists(QUOTA_STATE_FILE):
            with open(QUOTA_STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return (
                data.get("exhausted_month") == _current_year_month()
                and data.get("livekit_url") == current_url
            )
    except Exception:
        pass
    return False


def _mark_livekit_quota_exhausted_this_month() -> None:
    """Registra que el proyecto actual agotó sus créditos gratis este mes para enrutar directo a OpenRouter hasta el próximo mes."""
    ym = _current_year_month()
    current_url = os.environ.get("LIVEKIT_URL", "").strip()
    try:
        with open(QUOTA_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump({"exhausted_month": ym, "livekit_url": current_url}, f)
    except Exception:
        pass


class _QuotaCircuitBreakerHandler(logging.Handler):
    """Detecta en tiempo real el error 429 MaxGatewayCredits de LiveKit y activa el bypass mensual hacia OpenRouter."""

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if "MaxGatewayCredits" in msg or "QuotaStatusExceeded" in msg or "credit quota exceeded" in msg:
            _mark_livekit_quota_exhausted_this_month()


logging.getLogger("livekit.agents").addHandler(_QuotaCircuitBreakerHandler())

DAYS_ES = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
DAYS_EN = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS_ES = [
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
]
NUMBERS_ES = {
    1: "uno", 2: "dos", 3: "tres", 4: "cuatro", 5: "cinco",
    6: "seis", 7: "siete", 8: "ocho", 9: "nueve", 10: "diez",
    11: "once", 12: "doce", 13: "trece", 14: "catorce", 15: "quince",
    16: "dieciséis", 17: "diecisiete", 18: "dieciocho", 19: "diecinueve", 20: "veinte",
    21: "veintiuno", 22: "veintidós", 23: "veintitrés", 24: "veinticuatro", 25: "veinticinco",
    26: "veintiséis", 27: "veintisiete", 28: "veintiocho", 29: "veintinueve", 30: "treinta",
    31: "treinta y uno",
}
WEEKDAY_MAP = {
    "lunes": 0, "monday": 0,
    "martes": 1, "tuesday": 1,
    "miercoles": 2, "miércoles": 2, "wednesday": 2,
    "jueves": 3, "thursday": 3,
    "viernes": 4, "friday": 4,
    "sabado": 5, "sábado": 5, "saturday": 5,
    "domingo": 6, "sunday": 6,
}

PHONE_WORD_TO_DIGITS = {
    "cero": "0", "zero": "0", "oh": "0",
    "uno": "1", "un": "1", "una": "1", "one": "1",
    "dos": "2", "two": "2",
    "tres": "3", "three": "3",
    "cuatro": "4", "four": "4",
    "cinco": "5", "five": "5",
    "seis": "6", "six": "6",
    "siete": "7", "seven": "7",
    "ocho": "8", "eight": "8",
    "nueve": "9", "nine": "9",
    "diez": "10", "ten": "10",
    "once": "11", "eleven": "11",
    "doce": "12", "twelve": "12",
    "trece": "13", "thirteen": "13",
    "catorce": "14", "fourteen": "14",
    "quince": "15", "fifteen": "15",
    "dieciseis": "16", "dieciséis": "16", "sixteen": "16",
    "diecisiete": "17", "seventeen": "17",
    "dieciocho": "18", "eighteen": "18",
    "diecinueve": "19", "nineteen": "19",
    "veinte": "20", "twenty": "20",
    "veintiuno": "21", "veintiun": "21", "veintiún": "21",
    "veintidos": "22", "veintidós": "22",
    "veintitres": "23", "veintitrés": "23",
    "veinticuatro": "24",
    "veinticinco": "25",
    "veintiseis": "26", "veintiséis": "26",
    "veintisiete": "27",
    "veintiocho": "28",
    "veintinueve": "29",
}
PHONE_TENS_TO_INT = {
    "treinta": 30, "thirty": 30,
    "cuarenta": 40, "forty": 40,
    "cincuenta": 50, "fifty": 50,
    "sesenta": 60, "sixty": 60,
    "setenta": 70, "seventy": 70,
    "ochenta": 80, "eighty": 80,
    "noventa": 90, "ninety": 90,
}
PHONE_UNITS_TO_INT = {
    "uno": 1, "un": 1, "una": 1, "one": 1,
    "dos": 2, "two": 2,
    "tres": 3, "three": 3,
    "cuatro": 4, "four": 4,
    "cinco": 5, "five": 5,
    "seis": 6, "six": 6,
    "siete": 7, "seven": 7,
    "ocho": 8, "eight": 8,
    "nueve": 9, "nine": 9,
}
PHONE_HUNDREDS_TO_INT = {
    "cien": 100, "ciento": 100,
    "doscientos": 200, "doscientas": 200,
    "trescientos": 300, "trescientas": 300,
    "cuatrocientos": 400, "cuatrocientas": 400,
    "quinientos": 500, "quinientas": 500,
    "seiscientos": 600, "seiscientas": 600,
    "setecientos": 700, "setecientas": 700,
    "ochocientos": 800, "ochocientas": 800,
    "novecientos": 900, "novecientas": 900,
}
VE_AREA_PREFIXES = ("412", "414", "424", "416", "426", "422", "2")


def _format_to_e164_ve(raw_input: str) -> str:
    """Convierte cualquier teléfono venezolano (con o sin 0 inicial) al estándar internacional E.164 (+58...)."""
    digits = re.sub(r"\D", "", raw_input)
    if not digits:
        return ""
    if len(digits) == 11 and digits.startswith("0") and digits[1:].startswith(VE_AREA_PREFIXES):
        return f"+58{digits[1:]}"
    if len(digits) == 10 and digits.startswith(VE_AREA_PREFIXES):
        return f"+58{digits}"
    if len(digits) == 12 and digits.startswith("58") and digits[2:].startswith(VE_AREA_PREFIXES):
        return f"+{digits}"
    if len(digits) == 13 and digits.startswith("580") and digits[3:].startswith(VE_AREA_PREFIXES):
        return f"+58{digits[3:]}"
    if 10 <= len(digits) <= 15:
        return f"+{digits}"
    return ""


def _parse_sub_100(tokens: list[str], idx: int) -> tuple[int | None, int]:
    """Lee un número del 1 al 99 a partir de tokens[idx] para combinarlo con centenas o decenas."""
    if idx >= len(tokens):
        return None, idx
    tok = tokens[idx]
    if tok in PHONE_TENS_TO_INT:
        tens_val = PHONE_TENS_TO_INT[tok]
        if idx + 2 < len(tokens) and tokens[idx + 1] == "y" and tokens[idx + 2] in PHONE_UNITS_TO_INT:
            return tens_val + PHONE_UNITS_TO_INT[tokens[idx + 2]], idx + 3
        if idx + 1 < len(tokens) and tokens[idx + 1] in PHONE_UNITS_TO_INT:
            return tens_val + PHONE_UNITS_TO_INT[tokens[idx + 1]], idx + 2
        return tens_val, idx + 1
    if tok in PHONE_WORD_TO_DIGITS and int(PHONE_WORD_TO_DIGITS[tok]) >= 10:
        return int(PHONE_WORD_TO_DIGITS[tok]), idx + 1
    if tok in PHONE_UNITS_TO_INT:
        return PHONE_UNITS_TO_INT[tok], idx + 1
    return None, idx


def _extract_phone_digits_from_text(text: str) -> str:
    """Extrae de forma 100% determinista el teléfono en formato E.164 (+58...) aunque el STT agrupe en centenas o miles."""
    lower = text.lower()
    if "arroba" in lower or "@" in lower or "gmail" in lower or "hotmail" in lower or "outlook" in lower:
        return ""
    tokens = re.findall(r"[a-záéíóúñ]+|\d+", lower)
    result: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in ("mil", "millon", "millón", "millones"):
            i += 1
            continue
        if tok.isdigit():
            result.append(tok)
            i += 1
            continue
        if tok in PHONE_HUNDREDS_TO_INT:
            h_val = PHONE_HUNDREDS_TO_INT[tok]
            sub_val, next_i = _parse_sub_100(tokens, i + 1)
            if sub_val is not None:
                result.append(str(h_val + sub_val))
                i = next_i
            else:
                result.append(str(h_val))
                i += 1
            continue
        if tok in PHONE_TENS_TO_INT:
            sub_val, next_i = _parse_sub_100(tokens, i)
            if sub_val is not None:
                result.append(str(sub_val))
                i = next_i
                continue
        if tok in PHONE_WORD_TO_DIGITS:
            result.append(PHONE_WORD_TO_DIGITS[tok])
        i += 1

    return _format_to_e164_ve("".join(result))


def _format_date_phonetic(dt: datetime, language: str) -> str:
    """Convierte un objeto datetime a texto fonético exacto ('miércoles dieciocho de noviembre')."""
    if language == "es":
        day_name = DAYS_ES[dt.weekday()]
        day_word = NUMBERS_ES.get(dt.day, str(dt.day))
        month_name = MONTHS_ES[dt.month - 1]
        return f"{day_name} {day_word} de {month_name}"
    return dt.strftime("%A, %B %d")


def _get_time_of_day_phrases(language: str) -> tuple[str, str, str]:
    """Devuelve (periodo, saludo, deseo_despedida) acordes a la hora exacta de Venezuela (America/Caracas)."""
    now = datetime.now(ZoneInfo("America/Caracas"))
    if language == "es":
        if now.hour < 12:
            return "de la mañana", "buenos días", "un feliz día"
        if now.hour < 19:
            return "de la tarde", "buenas tardes", "una feliz tarde"
        return "de la noche", "buenas noches", "una feliz noche"
    if now.hour < 12:
        return "AM", "good morning", "a wonderful day"
    if now.hour < 19:
        return "PM", "good afternoon", "a wonderful afternoon"
    return "PM", "good evening", "a wonderful evening"


def _get_temporal_context(language: str) -> str:
    """Devuelve únicamente la fecha, día y hora actual exacta (America/Caracas) y el saludo correspondiente."""
    now = datetime.now(ZoneInfo("America/Caracas"))
    period, greeting_phrase, farewell_phrase = _get_time_of_day_phrases(language)
    if language == "es":
        day_name = DAYS_ES[now.weekday()]
        day_word = NUMBERS_ES.get(now.day, str(now.day))
        month_name = MONTHS_ES[now.month - 1]
        hour_12 = now.hour % 12 or 12
        return (
            f"Ahora mismo (now) es {day_name} {day_word} de {month_name} de {now.year} "
            f"(ISO: {now.strftime('%Y-%m-%d')}), {hour_12}:{now.minute:02d} {period}. "
            f"Saludo acorde a la hora actual: '{greeting_phrase}'. Despedida acorde a la hora actual: 'que pases {farewell_phrase}'. "
            "PROHIBIDO calcular fechas mentalmente: usa SIEMPRE la herramienta check_availability cuando el cliente mencione cualquier día o semana."
        )
    return (
        f"Right now (now) it is {now.strftime('%A, %B %d, %Y')} (ISO: {now.strftime('%Y-%m-%d')}), {now.strftime('%I:%M %p')}. "
        f"Time-appropriate greeting: '{greeting_phrase}'. Time-appropriate farewell: 'have {farewell_phrase}'. "
        "NEVER calculate dates mentally: ALWAYS use the check_availability tool whenever the prospect mentions any day or week."
    )


def _create_openrouter_client(model_name: str, api_key: str) -> llm.LLM:
    """Crea un cliente OpenRouter optimizado por latencia y blindado contra el bloqueo regional 403 de OpenAI."""
    provider_cfg: dict = {
        "sort": "latency",
        "allow_fallbacks": True,
    }
    if model_name.startswith("openai/"):
        provider_cfg["ignore"] = ["OpenAI"]

    try:
        return openai.LLM(
            model=model_name,
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            temperature=0.5,
            extra_body={"provider": provider_cfg},
        )
    except TypeError:
        return openai.LLM(
            model=model_name,
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            temperature=0.5,
        )


def _build_llm() -> llm.LLM:
    """Construye el motor LLM Multi-Tier: Primary = LiveKit Virginia (~400ms) → Fallback 1 = Claude Haiku 5.5 (~713ms) → Fallback 2 = Gemini 2.5 Flash (~711ms)."""
    llm_api_key = (
        os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    ).strip()

    openrouter_primary: llm.LLM | None = None
    openrouter_secondary: llm.LLM | None = None
    if llm_api_key:
        openrouter_primary = _create_openrouter_client(DEFAULT_OPENROUTER_LLM_MODEL, llm_api_key)
        if SECONDARY_OPENROUTER_LLM_MODEL and SECONDARY_OPENROUTER_LLM_MODEL != DEFAULT_OPENROUTER_LLM_MODEL:
            openrouter_secondary = _create_openrouter_client(SECONDARY_OPENROUTER_LLM_MODEL, llm_api_key)

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

    quota_exhausted = _is_livekit_quota_exhausted_this_month()
    if quota_exhausted and openrouter_primary is not None:
        fallback_chain = [m for m in (openrouter_primary, openrouter_secondary) if m is not None]
        if len(fallback_chain) > 1 and hasattr(llm, "FallbackAdapter"):
            for kwarg_name in ("llm", "llms"):
                try:
                    adapter = llm.FallbackAdapter(**{kwarg_name: fallback_chain})
                    logger.info(
                        "🧠 LiveKit quota exhausted for %s — routing directly to OpenRouter Dual Engine: Primary='%s' | Backup='%s'",
                        _current_year_month(),
                        DEFAULT_OPENROUTER_LLM_MODEL,
                        SECONDARY_OPENROUTER_LLM_MODEL,
                    )
                    return adapter
                except TypeError:
                    continue
        logger.info(
            "🧠 LiveKit quota exhausted for %s — routing directly to OpenRouter Primary='%s' (0ms retry overhead).",
            _current_year_month(),
            DEFAULT_OPENROUTER_LLM_MODEL,
        )
        return openrouter_primary

    if livekit_llm is not None and openrouter_primary is not None:
        chain = [m for m in (livekit_llm, openrouter_primary, openrouter_secondary) if m is not None]
        if hasattr(llm, "FallbackAdapter"):
            for kwarg_name in ("llm", "llms"):
                try:
                    adapter = llm.FallbackAdapter(**{kwarg_name: chain})
                    logger.info(
                        "🧠 Production Multi-Tier LLM active: Primary='%s' (LiveKit Virginia ~400ms) | Fallback 1='%s' (~713ms) | Fallback 2='%s' (~711ms)",
                        DEFAULT_LIVEKIT_LLM_MODEL,
                        DEFAULT_OPENROUTER_LLM_MODEL,
                        SECONDARY_OPENROUTER_LLM_MODEL,
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

    if openrouter_primary is not None:
        logger.info("🧠 Fallback LLM Engine active: '%s' (OpenRouter Lowest-Latency)", DEFAULT_OPENROUTER_LLM_MODEL)
        return openrouter_primary

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
    initial_wait: float = 1.2,
    max_wait: float = 18.0,
) -> None:
    """Monitorea en tiempo real el fin de la locución de despedida y corta la llamada sin cortar sílabas ni dejar silencios muertos."""
    try:
        session.input.set_audio_enabled(False)
    except Exception:
        pass

    # Dar margen para que el TTS inicie el playout de la frase de despedida
    await asyncio.sleep(initial_wait)

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
    await asyncio.sleep(0.7)
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
    voice_id: str | None = (
        metadata.get("voice_id")
        or os.environ.get("VOICE_ID")
        or "cartesia/sonic-2:5c5ad5e7-1020-476b-8b91-fdcbe9cc313c"
    )
    language: str = metadata.get("language") or "es"
    contact_name: str = metadata.get("contact_name", "Franlex")
    personalization_context: str = metadata.get(
        "personalization_context",
        "Cliente frecuente y leal de Zen Spa. Tiene pre-aprobado un cincuenta por ciento de descuento en depilación láser definitiva y valoración inicial sin costo.",
    )

    temporal_context = _get_temporal_context(language)
    _, current_greeting, current_farewell = _get_time_of_day_phrases(language)
    raw_business_instructions = agent_instructions.strip()
    raw_call_context = personalization_context.strip()
    client_label = contact_name.strip() if contact_name else ("Cliente" if language == "es" else "Prospect")

    if language == "es":
        agent_instructions = (
            "<runtime_data>\n"
            f"  <temporal_awareness>{temporal_context}</temporal_awareness>\n"
            "  <business_and_client_context>\n"
            f"    <client_name>{client_label}</client_name>\n"
            f"    <client_phone>{phone_number or 'este mismo número'}</client_phone>\n"
            f"    <call_context>{raw_call_context}</call_context>\n"
            f"    <business_instructions>{raw_business_instructions}</business_instructions>\n"
            "  </business_and_client_context>\n"
            "</runtime_data>\n\n"
            "<agent_system_manifest>\n"
            "  <identity>\n"
            f"    Eres {agent_display_name}, Senior Client Success Manager y especialista en ventas consultivas telefónicas. "
            "Tu tono es cálido, empático, natural y ejecutivo. Tratas al cliente de tú con cercanía sin repetir su nombre en cada turno.\n"
            "  </identity>\n\n"
            "  <execution_pipeline>\n"
            "    <phase index=\"1\" name=\"apertura_con_motivo_y_beneficio\">\n"
            f"      En cuanto el cliente confirme que es él después del 'Aló', saluda usando '{current_greeting}' (NUNCA digas 'buen día' si es de noche), "
            "menciona de dónde llamas, explica POR QUÉ lo llamas y cuál es su beneficio especial según <call_context> y <business_instructions>, y cierra preguntando suavemente si tiene medio minutito.\n"
            "    </phase>\n"
            "    <phase index=\"2\" name=\"explicacion_de_valor_antes_de_horarios\">\n"
            "      Cuando el cliente diga 'Sí, ok' o pregunte '¿De qué trata? / ¿Por qué me llamas?', NUNCA lances horarios en frío. "
            "Primero explica en 1 oración clara (máximo 25 palabras) en qué consiste el beneficio o paso propuesto en <business_instructions> sin compromiso, y cierra preguntando qué día o semana le quedaría más cómodo.\n"
            "    </phase>\n"
            "    <phase index=\"3\" name=\"consulta_determinista_de_fechas_y_horarios\">\n"
            "      - OBLIGATORIO: En cuanto el cliente mencione cualquier día o semana ('hoy', 'mañana', 'el jueves', 'el próximo viernes', 'la próxima semana'), invoca SIEMPRE la herramienta check_availability antes de responder. JAMÁS calcules fechas mentalmente.\n"
            "      - PROHIBIDO recitar una lista larga de horarios: ofrece SOLO DOS opciones concretas primero (una en la mañana y una en la tarde, ej. 'nueve de la mañana o tres de la tarde').\n"
            "      - Si el cliente pide otra hora, acéptala con flexibilidad. Si no puede en las horas sugeridas, empatiza y pregúntale '¿Qué hora te queda mejor a ti?'.\n"
            "    </phase>\n"
            "    <phase index=\"4\" name=\"confirmacion_paso_a_paso_whatsapp_y_correo\">\n"
            "      PROHIBIDO pedir WhatsApp y correo en la misma pregunta y PROHIBIDO colgar sin verificar los datos. Sigue este orden estricto de una pregunta a la vez:\n"
            "      - PASO 4.1 (Confirmar WhatsApp): Apenas el cliente elija día y hora, dile que le reservas ese horario, explícale que la confirmación se envía tanto por WhatsApp como por correo electrónico, y pregúntale ÚNICAMENTE: '¿Te envío la confirmación por WhatsApp a este mismo número?'.\n"
            "      - PASO 4.2 (Rama Sí / Rama No del WhatsApp):\n"
            "        * Si el cliente dice que SÍ usa este mismo número: pídele ahora su correo electrónico ('Perfecto, ¿y a qué correo electrónico te envío la confirmación?').\n"
            "        * Si el cliente dice que NO o que prefiere otro número: pregúntale primero '¿Cuál sería el mejor número telefónico para enviarte la confirmación por WhatsApp?'. Cuando te dé el número, repítelo tal como lo dijo el cliente en pares o dígitos sueltos (PROHIBIDO decir 'mil', 'ciento', 'cientos' o 'millones': si el transcriptor escribió 'cuatrocientos veinticuatro mil ciento veintidós seis siete nueve seis', tú debes decir 'cuatro veinticuatro, uno dos dos, seis siete nueve seis') y luego pídele su correo electrónico.\n"
            "      - PASO 4.3 (Verificación obligatoria del Correo / Read-Back): Cuando el cliente te dicte su correo electrónico, NO invoques book_appointment todavía. Repítele el correo en palabras fonéticas (escribiendo 'punto' y 'arroba' en letras para tu voz, ej. 'Perfecto, verifico que tu correo sea franlex punto ramirez dos arroba gmail punto com, ¿está correcto?'). Si el cliente dice que está incorrecto o corrige algo, haz la corrección y vuelve a repetírselo en palabras para confirmar.\n"
            "    </phase>\n"
            "    <phase index=\"5\" name=\"registro_final_y_despedida_acorde_a_la_hora\">\n"
            f"      - ÚNICAMENTE cuando el cliente confirme que el correo repetido SÍ está correcto, invoca OBLIGATORIAMENTE la herramienta book_appointment y dile que al finalizar la llamada estará recibiendo la confirmación tanto por WhatsApp como por correo, deseándole que pase {current_farewell}.\n"
            f"      - Si en cualquier momento el cliente rechaza explícitamente el servicio o la llamada ('no me interesa el servicio', 'estoy ocupado, adiós', 'número equivocado'), despídete deseándole {current_farewell} e invoca end_call.\n"
            "    </phase>\n"
            "  </execution_pipeline>\n\n"
            "  <hard_constraints>\n"
            f"    - MICRO-APERTURA TTS: Inicia siempre tu respuesta con una micro-frase de 2 a 3 palabras cerrada con PUNTO SEGUIDO ('Claro, {current_greeting}.', 'Perfecto, mira.', 'Claro, con gusto.') antes de la oración principal para acelerar tu voz.\n"
            "    - UNA PREGUNTA POR TURNO: Nunca hagas dos preguntas distintas en el mismo turno.\n"
            "    - FONÉTICA ESTRICTA Y TELÉFONOS NATURALES: Un solo párrafo continuo sin saltos de línea. PROHIBIDO usar signos de exclamación (¡ !) o puntos suspensivos al inicio. Escribe TODOS los números, fechas, porcentajes y horas en letras minúsculas. JAMÁS pronuncies las palabras 'mil', 'ciento', 'cientos' ni 'millones' al repetir un número telefónico.\n"
            "    - AISLAMIENTO DE HERRAMIENTAS: JAMÁS escribas ni pronuncies nombres de funciones o código en tu respuesta de texto.\n"
            "  </hard_constraints>\n"
            "</agent_system_manifest>"
        )
    else:
        agent_instructions = (
            "<runtime_data>\n"
            f"  <temporal_awareness>{temporal_context}</temporal_awareness>\n"
            "  <business_and_client_context>\n"
            f"    <client_name>{client_label}</client_name>\n"
            f"    <client_phone>{phone_number or 'this same number'}</client_phone>\n"
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
            f"      As soon as the prospect confirms it is them after 'Hello?', greet them using '{current_greeting}', "
            "state where you are calling from, explain WHY you are calling and their specific benefit based on <call_context> and <business_instructions>, and softly ask if they have half a minute.\n"
            "    </phase>\n"
            "    <phase index=\"2\" name=\"value_explanation_before_scheduling\">\n"
            "      When the prospect says 'Sure / Okay' or asks 'What is this about? / Why is it free?', NEVER push time slots cold. "
            "First explain in 1 clear sentence (max 25 words) how the benefit or next step in <business_instructions> works with no obligation, then ask what day or week works best for them.\n"
            "    </phase>\n"
            "    <phase index=\"3\" name=\"deterministic_date_lookup_and_negotiation\">\n"
            "      - MANDATORY: Whenever the prospect mentions any day or relative week ('today', 'tomorrow', 'this Thursday', 'next Friday'), ALWAYS invoke the check_availability tool first. NEVER calculate dates mentally.\n"
            "      - NEVER recite a long list of times: offer ONLY TWO concrete options first (one in the morning and one in the afternoon).\n"
            "      - If the prospect suggests another time, accept it immediately. If neither works, empathize and ask 'What time works best for you?'.\n"
            "    </phase>\n"
            "    <phase index=\"4\" name=\"step_by_step_whatsapp_and_email_verification\">\n"
            "      NEVER ask for WhatsApp and email in the same question, and NEVER hang up without verifying their email. Follow this strict one-question-at-a-time sequence:\n"
            "      - STEP 4.1 (Confirm WhatsApp): Once day and time are agreed, tell them you are reserving that slot, explain that the confirmation is sent via both WhatsApp and email, and ask ONLY: 'Should I send the WhatsApp confirmation to this same phone number?'.\n"
            "      - STEP 4.2 (WhatsApp Yes / No Branch):\n"
            "        * If they say YES to this same number: ask for their email address ('Perfect, and what email address should I send the confirmation to?').\n"
            "        * If they say NO or prefer another number: first ask 'What is the best phone number to send your WhatsApp confirmation to?'. Once they give the number, repeat it back naturally in individual digits or pairs (NEVER say 'thousand' or 'hundred') and then ask for their email address.\n"
            "      - STEP 4.3 (Mandatory Email Read-Back Loop): When the prospect gives their email address, DO NOT invoke book_appointment yet. Repeat the email back clearly in spoken words (using 'dot' and 'at' for your voice) and ask 'Is that email correct?'. If they say it is incorrect, apply the correction and repeat it back again for confirmation.\n"
            "    </phase>\n"
            "    <phase index=\"5\" name=\"final_booking_and_time_aware_goodbye\">\n"
            f"      - ONLY after the prospect confirms that the repeated email is YES correct, MANDATORILY invoke the book_appointment tool and let them know that as soon as the call ends they will receive the confirmation via both WhatsApp and email, wishing them {current_farewell}.\n"
            f"      - If at any point the prospect explicitly rejects the service or call ('not interested', 'too busy, bye', 'wrong number'), wish them {current_farewell} and invoke end_call.\n"
            "    </phase>\n"
            "  </execution_pipeline>\n\n"
            "  <hard_constraints>\n"
            f"    - TTS MICRO-OPENER: Always start your turn with a 2-to-3 word micro-phrase ending in a PERIOD ('Sure, {current_greeting}.', 'Perfect, so look.', 'Got it, sure.') before your main sentence.\n"
            "    - ONE QUESTION PER TURN: Never ask two different questions in the same turn.\n"
            "    - STRICT PHONETICS: Reply in a single continuous paragraph without line breaks. NEVER use exclamation marks (!) or leading ellipses (...). Spell out ALL numbers, phone digits, dates, percentages, and times in lowercase words. NEVER read phone numbers as 'hundreds' or 'thousands'.\n"
            "    - TOOL ISOLATION: NEVER write or speak function names or code in your text response.\n"
            "  </hard_constraints>\n"
            "</agent_system_manifest>"
        )

    logger.info(
        "Starting call as '%s' (voice=%s, language=%s)",
        agent_display_name, voice_id or DEFAULT_TTS_MODEL, language,
    )

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Parsear "provider/model:voice-id" y normalizar parámetros según proveedor (Cartesia, Deepgram, Rime)
    model = voice_id if voice_id else DEFAULT_TTS_MODEL
    voice: str | None = None
    if voice_id and ":" in voice_id:
        idx = voice_id.rfind(":")
        voice = voice_id[idx + 1:]
        model = voice_id[:idx]

    if model.startswith("deepgram/") and voice:
        voice = re.sub(r"-(es|en)$", "", voice)
    elif model.startswith("rime/") and voice and voice.lower() == "serena":
        model = "rime/arcana"

    tts_kwargs: dict = {"model": model}
    if voice:
        tts_kwargs["voice"] = voice
    if not model.startswith("rime/"):
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
    booking_state: dict = {
        "last_user_text": "",
        "email_read_back_to_user": False,
        "extracted_phone_digits": "",
    }

    @llm.function_tool(
        description=(
            "Calcula en Python la fecha exacta del calendario y consulta los horarios disponibles. "
            "Invócala SIEMPRE que el cliente mencione un día ('hoy', 'mañana', 'lunes', 'miércoles', 'jueves', 'viernes', 'sábado') "
            "o una semana ('esta semana'=0, 'próxima semana' o 'semana entrante'=1, 'en dos semanas'=2, 'en tres semanas'=3)."
        )
    )
    async def check_availability(
        day_of_week: str,
        weeks_ahead: int = 0,
    ) -> str:
        now = datetime.now(ZoneInfo("America/Caracas"))
        clean_day = day_of_week.lower().strip()
        last_u_day = booking_state["last_user_text"].lower()

        if "pasado mañana" in clean_day or "day after tomorrow" in clean_day:
            target_dt = now + timedelta(days=2)
            phonetic = _format_date_phonetic(target_dt, language)
            return (
                f"Fecha exacta calculada para pasado mañana: {phonetic} (ISO: {target_dt.strftime('%Y-%m-%d')}). "
                "INSTRUCCIÓN: Ofrece primero SOLO DOS opciones (nueve de la mañana o tres de la tarde). También tienes libre diez y once de la mañana, y cuatro y cinco de la tarde."
            )

        if "mañana" in clean_day or "tomorrow" in clean_day:
            target_dt = now + timedelta(days=1)
            phonetic = _format_date_phonetic(target_dt, language)
            return (
                f"Fecha exacta calculada para mañana: {phonetic} (ISO: {target_dt.strftime('%Y-%m-%d')}). "
                "INSTRUCCIÓN: Ofrece primero SOLO DOS opciones (nueve de la mañana o tres de la tarde). También tienes libre diez y once de la mañana, y cuatro y cinco de la tarde."
            )

        if "hoy" in clean_day or "today" in clean_day or "para hoy" in last_u_day or "es hoy" in last_u_day:
            phonetic = _format_date_phonetic(now, language)
            logger.info("📅 [CALENDAR TOOL] day='%s' (HOY) -> %s (%s)", day_of_week, phonetic, now.strftime("%Y-%m-%d"))
            if language == "es":
                return (
                    f"Fecha exacta calculada para HOY: {phonetic} (ISO: {now.strftime('%Y-%m-%d')}). "
                    "INSTRUCCIÓN: Sí tienes disponibilidad para hoy. Ofrece primero SOLO DOS opciones (por ejemplo: once de la mañana o tres de la tarde). También tienes libre cuatro y cinco de la tarde."
                )
            return (
                f"Exact date for TODAY: {phonetic} (ISO: {now.strftime('%Y-%m-%d')}). "
                "INSTRUCTION: You DO have availability today. Offer ONLY TWO options first (e.g., eleven in the morning or three in the afternoon)."
            )

        matched_weekday: int | None = None
        for key, idx in WEEKDAY_MAP.items():
            if key in clean_day:
                matched_weekday = idx
                break

        if matched_weekday is None:
            matched_weekday = now.weekday()

        # Lunes de la semana actual como ancla matemática exacta
        monday_current_week = now - timedelta(days=now.weekday())
        this_week_dt = monday_current_week + timedelta(days=matched_weekday)
        next_week_dt = monday_current_week + timedelta(weeks=1, days=matched_weekday)
        two_weeks_dt = monday_current_week + timedelta(weeks=2, days=matched_weekday)

        effective_weeks = max(0, int(weeks_ahead))
        if effective_weeks == 0 and this_week_dt.date() < now.date():
            effective_weeks = 1

        requested_dt = monday_current_week + timedelta(weeks=effective_weeks, days=matched_weekday)

        req_phonetic = _format_date_phonetic(requested_dt, language)
        this_wk_phonetic = _format_date_phonetic(this_week_dt, language)
        next_wk_phonetic = _format_date_phonetic(next_week_dt, language)
        two_wk_phonetic = _format_date_phonetic(two_weeks_dt, language)

        logger.info(
            "📅 [CALENDAR TOOL] day='%s' weeks_ahead=%d -> %s (%s)",
            day_of_week,
            effective_weeks,
            req_phonetic,
            requested_dt.strftime("%Y-%m-%d"),
        )

        if language == "es":
            return (
                f"RESULTADO EXACTO DE CALENDARIO PYTHON:\n"
                f"- Fecha solicitada ({effective_weeks} semanas adelante): {req_phonetic} ({requested_dt.strftime('%Y-%m-%d')}).\n"
                f"- Desglose de seguridad: En esta semana actual es {this_wk_phonetic}; "
                f"en la próxima semana (semana entrante) es {next_wk_phonetic}; "
                f"en dos semanas es {two_wk_phonetic}.\n"
                f"- INSTRUCCIÓN DE RESPUESTA: NO leas todos los horarios disponibles. Ofrece primero SOLO DOS opciones (por ejemplo: nueve de la mañana o tres de la tarde). Si el cliente pide otra hora, también tienes libre diez y once de la mañana, y cuatro y cinco de la tarde."
            )
        return (
            f"EXACT PYTHON CALENDAR RESULT:\n"
            f"- Requested date ({effective_weeks} weeks ahead): {req_phonetic} ({requested_dt.strftime('%Y-%m-%d')}).\n"
            f"- Reference breakdown: This week is {this_wk_phonetic}; next week is {next_wk_phonetic}; in two weeks is {two_wk_phonetic}.\n"
            f"- RESPONSE INSTRUCTION: DO NOT read all available slots out loud. Offer ONLY TWO options first (e.g., nine in the morning or three in the afternoon). If they ask for another time, ten am, eleven am, four pm, and five pm are also open."
        )

    @llm.function_tool(
        description=(
            "Registra formalmente la cita ÚNICAMENTE DESPUÉS de haberle repetido el correo electrónico al cliente "
            "y que el cliente haya respondido que SÍ está correcto. NUNCA la invoques en el mismo turno donde el cliente apenas dicta su correo."
        )
    )
    async def book_appointment(
        appointment_date: str,
        appointment_time: str,
        whatsapp_number: str = "mismo número de la llamada",
        email: str = "no proporcionado",
    ) -> str:
        nonlocal call_ending
        last_u = booking_state["last_user_text"].lower()

        # Candado determinista: si el cliente acaba de dictar el correo o pidió otro número y aún no se le ha repetido el correo, bloquear cierre prematuro
        just_dictated_email_or_phone_change = (
            "arroba" in last_u
            or "@" in last_u
            or "gmail" in last_u
            or "hotmail" in last_u
            or "outlook" in last_u
            or "otro número" in last_u
            or "otro numero" in last_u
            or "different number" in last_u
        )
        if not booking_state["email_read_back_to_user"] and just_dictated_email_or_phone_change:
            booking_state["email_read_back_to_user"] = True
            logger.info(
                "🛡️ [GUARDRAIL] Blocked premature book_appointment — forcing agent to verify phone/email ('%s') with user first.",
                email,
            )
            if "otro número" in last_u or "otro numero" in last_u:
                return (
                    f"ALTO - NO CIERRES LA LLAMADA TODAVÍA: El cliente acaba de indicar que prefiere usar OTRO número telefónico para WhatsApp "
                    f"(y mencionó el correo '{email}'). Pregúntale primero cuál es el mejor número telefónico para su WhatsApp y luego confírmale el correo '{email}' preguntando si está correcto."
                )
            return (
                f"ALTO - NO CIERRES LA LLAMADA TODAVÍA: El cliente acaba de dictar su correo ('{email}'). "
                f"Según el protocolo obligatorio, primero debes repetirle textualmente el correo '{email}' en palabras y preguntarle: '¿Está correcto tu correo?'. "
                "Solo cuando el cliente te responda que sí, vuelve a invocar book_appointment."
            )

        llm_e164 = _format_to_e164_ve(whatsapp_number)
        if booking_state["extracted_phone_digits"]:
            resolved_phone = booking_state["extracted_phone_digits"]
            logger.info(
                "📞 [PHONE PARSER] Deterministic Python E.164 phone applied: '%s' (LLM passed: '%s')",
                resolved_phone,
                whatsapp_number,
            )
        elif llm_e164:
            resolved_phone = llm_e164
        elif ("mismo" in whatsapp_number.lower() or "same" in whatsapp_number.lower()) and phone_number:
            resolved_phone = _format_to_e164_ve(phone_number) or phone_number
        else:
            resolved_phone = whatsapp_number

        booking_payload = {
            "client_name": contact_name or "Cliente",
            "appointment_date": appointment_date,
            "appointment_time": appointment_time,
            "whatsapp_number": resolved_phone,
            "email": email,
            "booked_at_iso": datetime.now(ZoneInfo("America/Caracas")).isoformat(),
        }
        logger.info("✅ [BOOKING CONFIRMED] %s", json.dumps(booking_payload, ensure_ascii=False))

        if not call_ending:
            call_ending = True
            logger.info("📴 Booking completed — waiting for full confirmation & goodbye audio before hanging up…")
            asyncio.create_task(_terminate_call_session(ctx, session, initial_wait=2.5, max_wait=18.0))

        _, _, dynamic_farewell = _get_time_of_day_phrases(language)
        if language == "es":
            return (
                f"Cita confirmada exitosamente para el {appointment_date} a las {appointment_time} "
                f"(WhatsApp: {resolved_phone}, Correo confirmado: {email}). "
                f"Ahora dile al cliente en una frase cálida que al finalizar la llamada estará recibiendo la confirmación tanto por WhatsApp como por correo electrónico, "
                f"y despídete deseándole que pase {dynamic_farewell}."
            )
        return (
            f"Appointment confirmed for {appointment_date} at {appointment_time} "
            f"(WhatsApp: {resolved_phone}, Confirmed Email: {email}). "
            f"Now tell the prospect warmly that upon ending the call they will receive the confirmation via both WhatsApp and email, "
            f"and wish them {dynamic_farewell}."
        )

    @llm.function_tool(
        description=(
            "Cuelga y finaliza la llamada telefónica. Invócala cuando el cliente rechace el servicio/llamada o después de despedirte."
        )
    )
    async def end_call() -> None:
        nonlocal call_ending
        if call_ending:
            return None
        call_ending = True
        logger.info("📴 Agent triggered end_call tool — waiting for goodbye audio to finish before hanging up…")
        asyncio.create_task(_terminate_call_session(ctx, session, initial_wait=1.2, max_wait=18.0))
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
        tools=[check_availability, book_appointment, end_call],
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
            booking_state["last_user_text"] = text
            parsed_digits = _extract_phone_digits_from_text(text)
            if parsed_digits:
                booking_state["extracted_phone_digits"] = parsed_digits
                logger.info("📞 [PHONE PARSER] Captured exact E.164 phone from user speech: %s", parsed_digits)
            print(f"\n👤 [{contact_name or 'Usuario'}]: {text}", flush=True)
        elif role == "assistant":
            clean_text = text.replace("functions.end_call", "").replace("end_call()", "").strip()
            if "arroba" in clean_text.lower() or "@" in clean_text.lower() or "correo es" in clean_text.lower():
                booking_state["email_read_back_to_user"] = True
            print(f"\n🤖 [{agent_display_name}]: {clean_text}\n", flush=True)
            if not call_ending and ("functions.end_call" in text or "end_call(" in text):
                call_ending = True
                logger.info("📴 Safety-net triggered end_call from assistant text — waiting for goodbye audio…")
                asyncio.create_task(_terminate_call_session(ctx, session, initial_wait=1.2, max_wait=18.0))

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
            asyncio.create_task(_terminate_call_session(ctx, session, initial_wait=0.1, max_wait=0.2))

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