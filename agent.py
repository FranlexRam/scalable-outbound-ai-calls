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
    stt as lk_stt,
    tts as lk_tts,
)
from livekit.agents.inference import (
    LLM as InferenceLLM,
    STT as InferenceSTT,
    TTS as InferenceTTS,
)
from livekit.agents.voice import Agent
from livekit.agents.voice.room_io import RoomOptions
from livekit.plugins import cartesia, deepgram, openai, silero

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
DEFAULT_ES_TEST_VOICE = "deepgram/aura-2:celeste-es"
DEFAULT_EN_TEST_VOICE = "cartesia/sonic-3:f786b574-daa5-4673-aa0c-cbe3e8534c02"
DEFAULT_LIVEKIT_LLM_MODEL = os.environ.get("LIVEKIT_LLM_MODEL", "openai/gpt-4o-mini")
DEFAULT_OPENROUTER_LLM_MODEL = os.environ.get("OPENROUTER_LLM_MODEL", "google/gemini-2.5-flash")
SECONDARY_OPENROUTER_LLM_MODEL = os.environ.get("SECONDARY_OPENROUTER_LLM_MODEL", "anthropic/claude-haiku-5.5")
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
        if (
            "MaxGatewayCredits" in msg
            or "QuotaStatusExceeded" in msg
            or "credit quota exceeded" in msg
            or ("agent-gateway.livekit.cloud" in msg and "429" in msg)
        ):
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
NUMBERS_ORDINAL_EN = {
    1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
    6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth",
    11: "eleventh", 12: "twelfth", 13: "thirteenth", 14: "fourteenth", 15: "fifteenth",
    16: "sixteenth", 17: "seventeenth", 18: "eighteenth", 19: "nineteenth", 20: "twentieth",
    21: "twenty-first", 22: "twenty-second", 23: "twenty-third", 24: "twenty-fourth", 25: "twenty-fifth",
    26: "twenty-sixth", 27: "twenty-seventh", 28: "twenty-eighth", 29: "twenty-ninth", 30: "thirtieth",
    31: "thirty-first",
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
    "cien": 100, "ciento": 100, "hundred": 100,
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

MONTH_NAME_TO_INT = {
    "enero": 1, "january": 1,
    "febrero": 2, "february": 2,
    "marzo": 3, "march": 3,
    "abril": 4, "april": 4,
    "mayo": 5, "may": 5,
    "junio": 6, "june": 6,
    "julio": 7, "july": 7,
    "agosto": 8, "august": 8,
    "septiembre": 9, "september": 9,
    "octubre": 10, "october": 10,
    "noviembre": 11, "november": 11,
    "diciembre": 12, "december": 12,
}
DAY_WORD_TO_INT = {
    **{v: k for k, v in NUMBERS_ES.items()},
    **{v: k for k, v in NUMBERS_ORDINAL_EN.items()},
    "primero": 1, "primer": 1, "veintiun": 21, "veintiún": 21,
    "veintidos": 22, "veintitres": 23, "veintiseis": 26, "dieciseis": 16,
}
HOUR_WORD_TO_INT = {
    "una": 1, "uno": 1, "one": 1,
    "dos": 2, "two": 2,
    "tres": 3, "three": 3,
    "cuatro": 4, "four": 4,
    "cinco": 5, "five": 5,
    "seis": 6, "six": 6,
    "siete": 7, "seven": 7,
    "ocho": 8, "eight": 8,
    "nueve": 9, "nine": 9,
    "diez": 10, "ten": 10,
    "once": 11, "eleven": 11,
    "doce": 12, "twelve": 12,
}
HOURS_ES_WORDS = {
    1: "una", 2: "dos", 3: "tres", 4: "cuatro", 5: "cinco", 6: "seis",
    7: "siete", 8: "ocho", 9: "nueve", 10: "diez", 11: "once", 12: "doce",
}
HOURS_EN_WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
    7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve",
}


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
    """Extrae de forma 100% determinista el teléfono en formato E.164 (+58...) en español o inglés (incluyendo 'double'/'triple')."""
    lower = text.lower()
    if (
        "arroba" in lower
        or "@" in lower
        or "gmail" in lower
        or "hotmail" in lower
        or "outlook" in lower
        or "yahoo" in lower
        or "icloud" in lower
        or "dot com" in lower
    ):
        return ""
    tokens = re.findall(r"[a-záéíóúñ]+|\d+", lower)
    result: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in ("mil", "millon", "millón", "millones", "thousand", "million"):
            i += 1
            continue
        if tok in ("double", "doble") and i + 1 < len(tokens):
            nxt = tokens[i + 1]
            d = nxt if nxt.isdigit() else PHONE_WORD_TO_DIGITS.get(nxt, "")
            if len(d) == 1:
                result.append(d * 2)
                i += 2
                continue
        if tok in ("triple",) and i + 1 < len(tokens):
            nxt = tokens[i + 1]
            d = nxt if nxt.isdigit() else PHONE_WORD_TO_DIGITS.get(nxt, "")
            if len(d) == 1:
                result.append(d * 3)
                i += 2
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


def _sanitize_stt_text(text: str, contact_name: str = "") -> str:
    """Corrige errores fonéticos recurrentes del STT en español e inglés (ej. 'Franklex' -> 'Franlex')."""
    lower = text.lower()
    asks_to_remove_k = (
        "sin la k" in lower
        or "sin k" in lower
        or "pusiste una k" in lower
        or "agregaste una k" in lower
        or "letra k" in lower
        or "no lleva k" in lower
        or "without the k" in lower
        or "without k" in lower
        or "no k" in lower
        or "letter k" in lower
    )
    if asks_to_remove_k or contact_name.strip().lower() == "franlex":
        if asks_to_remove_k:
            text = re.sub(r"(?i)\bfranklex\b", "franlex", text)
    return text


def _format_date_phonetic(dt: datetime, language: str) -> str:
    """Convierte un objeto datetime a texto fonético exacto sin el año en español o inglés."""
    if language == "es":
        day_name = DAYS_ES[dt.weekday()]
        day_word = NUMBERS_ES.get(dt.day, str(dt.day))
        month_name = MONTHS_ES[dt.month - 1]
        return f"{day_name} {day_word} de {month_name}"
    day_name_en = DAYS_EN[dt.weekday()]
    month_name_en = dt.strftime("%B")
    day_ordinal_en = NUMBERS_ORDINAL_EN.get(dt.day, str(dt.day))
    return f"{day_name_en}, {month_name_en} {day_ordinal_en}"


def _normalize_booking_datetime(
    raw_date: str,
    raw_time: str,
    language: str,
    fallback_iso_date: str = "",
) -> tuple[str, str, str, str, str]:
    """
    Normaliza cualquier fecha y hora (ISO o en letras ES/EN) para Supabase y para TTS:
    Devuelve: (iso_date 'YYYY-MM-DD', iso_time 'HH:MM:SS', iso_timestamptz, spoken_date_no_year, spoken_time_words)
    """
    tz = ZoneInfo("America/Caracas")
    now = datetime.now(tz)
    clean_d = raw_date.strip()
    target_dt: datetime | None = None

    # 1. Intentar parsear si viene en formato ISO YYYY-MM-DD
    iso_match = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", clean_d)
    if iso_match:
        try:
            target_dt = datetime(
                int(iso_match.group(1)),
                int(iso_match.group(2)),
                int(iso_match.group(3)),
                tzinfo=tz,
            )
        except ValueError:
            target_dt = None

    # 2. Si viene en letras ("jueves tres de diciembre", "Thursday, December third"), extraer mes y día
    if target_dt is None:
        lower_d = re.sub(
            r"(?i)\s*(de\s+|del\s+|in\s+|of\s+)?(dos\s+mil\s+veint[a-záéíóú]+|two\s+thousand\s+twenty[\s-]*\w+|twenty\s+twenty[\s-]*\w+|202\d)\b",
            "",
            clean_d.lower(),
        )
        detected_month: int | None = None
        for m_name, m_val in MONTH_NAME_TO_INT.items():
            if m_name in lower_d:
                detected_month = m_val
                break

        detected_day: int | None = None
        if "treinta y uno" in lower_d or "thirty-first" in lower_d:
            detected_day = 31
        else:
            num_match = re.search(r"\b([12]?\d|3[01])(?:st|nd|rd|th)?\b", lower_d)
            if num_match:
                detected_day = int(num_match.group(1))
            else:
                for token in re.findall(r"[a-záéíóúñ-]+", lower_d):
                    if token in DAY_WORD_TO_INT:
                        detected_day = DAY_WORD_TO_INT[token]
                        break

        if detected_month and detected_day:
            year = now.year
            if detected_month < now.month or (detected_month == now.month and detected_day < now.day):
                year += 1
            try:
                target_dt = datetime(year, detected_month, detected_day, tzinfo=tz)
            except ValueError:
                target_dt = None

    # 3. Respaldo con la última fecha consultada en check_availability o fecha siguiente
    if target_dt is None and fallback_iso_date:
        try:
            parsed_fb = datetime.strptime(fallback_iso_date, "%Y-%m-%d")
            target_dt = datetime(parsed_fb.year, parsed_fb.month, parsed_fb.day, tzinfo=tz)
        except ValueError:
            target_dt = None

    if target_dt is None:
        target_dt = now + timedelta(days=1)

    # 4. Parsear la hora ("11:00", "17:00", "once de la mañana", "three in the afternoon")
    lower_t = raw_time.strip().lower()
    hour_24 = 9
    minute_val = 0

    hhmm_match = re.search(r"(\d{1,2}):(\d{2})", lower_t)
    if hhmm_match:
        hour_24 = int(hhmm_match.group(1))
        minute_val = int(hhmm_match.group(2))
    else:
        digit_h = re.search(r"\b(\d{1,2})\b", lower_t)
        if digit_h:
            hour_24 = int(digit_h.group(1))
        else:
            for tok in re.findall(r"[a-záéíóúñ]+", lower_t):
                if tok in HOUR_WORD_TO_INT:
                    hour_24 = HOUR_WORD_TO_INT[tok]
                    break

    is_pm = any(k in lower_t for k in ("pm", "p.m", "tarde", "noche", "afternoon", "evening"))
    is_am = any(k in lower_t for k in ("am", "a.m", "mañana", "morning"))
    if is_pm and hour_24 < 12:
        hour_24 += 12
    elif is_am and hour_24 == 12:
        hour_24 = 0
    elif not is_am and not is_pm and 1 <= hour_24 <= 6:
        hour_24 += 12

    combined_dt = datetime(
        target_dt.year,
        target_dt.month,
        target_dt.day,
        hour_24,
        minute_val,
        0,
        tzinfo=tz,
    )
    iso_date_str = combined_dt.strftime("%Y-%m-%d")
    iso_time_str = combined_dt.strftime("%H:%M:%S")
    iso_full_str = combined_dt.isoformat()

    spoken_date_str = _format_date_phonetic(combined_dt, language)
    hour_12 = hour_24 % 12 or 12
    if language == "es":
        period_es = "de la mañana" if hour_24 < 12 else ("de la tarde" if hour_24 < 19 else "de la noche")
        spoken_time_str = f"{HOURS_ES_WORDS.get(hour_12, str(hour_12))} {period_es}"
    else:
        period_en = "in the morning" if hour_24 < 12 else ("in the afternoon" if hour_24 < 19 else "in the evening")
        spoken_time_str = f"{HOURS_EN_WORDS.get(hour_12, str(hour_12))} {period_en}"

    return iso_date_str, iso_time_str, iso_full_str, spoken_date_str, spoken_time_str


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
        return "in the morning", "good morning", "a wonderful day"
    if now.hour < 19:
        return "in the afternoon", "good afternoon", "a wonderful afternoon"
    return "in the evening", "good evening", "a wonderful evening"


def _get_temporal_context(language: str, custom_schedule: str = "") -> str:
    """Pre-calcula en Python el calendario exacto de 14 días (en ES y EN) e inyecta el horario configurable del negocio."""
    now = datetime.now(ZoneInfo("America/Caracas"))
    period, greeting_phrase, farewell_phrase = _get_time_of_day_phrases(language)
    monday_this_week = now - timedelta(days=now.weekday())
    tomorrow = now + timedelta(days=1)
    day_after = now + timedelta(days=2)

    this_week_items = []
    next_week_items = []
    for d_idx in range(7):
        dt_this = monday_this_week + timedelta(days=d_idx)
        dt_next = monday_this_week + timedelta(weeks=1, days=d_idx)
        if dt_this.date() >= now.date():
            this_week_items.append(f"{_format_date_phonetic(dt_this, language)} [ISO: {dt_this.strftime('%Y-%m-%d')}]")
        next_week_items.append(f"{_format_date_phonetic(dt_next, language)} [ISO: {dt_next.strftime('%Y-%m-%d')}]")

    hour_12 = now.hour % 12 or 12

    if language == "es":
        schedule_es = custom_schedule.strip() or (
            "Horario de atención del negocio: de nueve de la mañana a cinco de la tarde. "
            "Cupos disponibles actualmente todos los días: en la mañana nueve, diez y once de la mañana; "
            "en la tarde tres, cuatro y cinco de la tarde."
        )
        return (
            f"Ahora mismo (HOY) es {_format_date_phonetic(now, 'es')} [ISO: {now.strftime('%Y-%m-%d')}], {hour_12}:{now.minute:02d} {period}. "
            f"MAÑANA es {_format_date_phonetic(tomorrow, 'es')} [ISO: {tomorrow.strftime('%Y-%m-%d')}]. "
            f"PASADO MAÑANA es {_format_date_phonetic(day_after, 'es')} [ISO: {day_after.strftime('%Y-%m-%d')}]. "
            f"Días restantes de ESTA SEMANA: {'; '.join(this_week_items)}. "
            f"Días de LA PRÓXIMA SEMANA (semana entrante): {'; '.join(next_week_items)}. "
            f"{schedule_es} "
            f"Saludo actual: '{greeting_phrase}'. Despedida actual: 'que pases {farewell_phrase}'."
        )

    schedule_en = custom_schedule.strip() or (
        "Business operating hours: nine in the morning to five in the afternoon. "
        "Currently available slots every day: in the morning at nine, ten, and eleven in the morning; "
        "in the afternoon at three, four, and five in the afternoon."
    )
    return (
        f"Right now (TODAY) is {_format_date_phonetic(now, 'en')} [ISO: {now.strftime('%Y-%m-%d')}], {hour_12}:{now.minute:02d} {period}. "
        f"TOMORROW is {_format_date_phonetic(tomorrow, 'en')} [ISO: {tomorrow.strftime('%Y-%m-%d')}]. "
        f"DAY AFTER TOMORROW is {_format_date_phonetic(day_after, 'en')} [ISO: {day_after.strftime('%Y-%m-%d')}]. "
        f"Remaining days THIS WEEK: {'; '.join(this_week_items)}. "
        f"Days NEXT WEEK: {'; '.join(next_week_items)}. "
        f"{schedule_en} "
        f"Current greeting: '{greeting_phrase}'. Current farewell: 'have {farewell_phrase}'."
    )


def _create_openrouter_client(model_name: str, api_key: str) -> llm.LLM:
    """Crea un cliente OpenRouter optimizado por latencia (con modo conversacional directo ~756ms para Gemini 2.5 Flash)."""
    provider_cfg: dict = {
        "sort": "latency",
        "allow_fallbacks": True,
    }
    if model_name.startswith("anthropic/"):
        provider_cfg["order"] = ["Anthropic"]

    extra_body: dict = {"provider": provider_cfg}
    if "gemini-2.5" in model_name:
        extra_body["reasoning"] = {"max_tokens": 0}

    try:
        return openai.LLM(
            model=model_name,
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            temperature=0.5,
            extra_body=extra_body,
        )
    except TypeError:
        return openai.LLM(
            model=model_name,
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            temperature=0.5,
        )


def _build_llm() -> llm.LLM:
    """Construye el motor LLM Multi-Tier: Primary = LiveKit Virginia (~400ms) → Fallback 1 = Gemini 2.5 Flash (~756ms) → Fallback 2 = Claude Haiku 5.5."""
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
                        "🧠 Production Multi-Tier LLM active: Primary='%s' (LiveKit Virginia ~400ms) | Fallback 1='%s' (~756ms) | Fallback 2='%s'",
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


def _build_stt(language: str, contact_name: str = "") -> lk_stt.STT:
    """Construye el motor STT (Oído) con Deepgram Directo (BYOK ~150ms) incluyendo muletillas humanas y Cartesia STT como respaldo."""
    deepgram_api_key = os.environ.get("DEEPGRAM_API_KEY", "").strip()
    cartesia_api_key = os.environ.get("CARTESIA_API_KEY", "").strip()
    stt_lang = "es" if language == "es" else "en"

    primary_stt: lk_stt.STT | None = None
    backup_stt: lk_stt.STT | None = None

    if deepgram_api_key:
        custom_terms = [
            "Franlex",
            "Ramírez",
            "Zen Spa",
            "Plexus Code",
            "gmail.com",
            "mmm",
            "eh",
            "ajá",
            "este",
            "uh",
            "um",
            "hmm",
        ]
        if contact_name and contact_name.strip() not in custom_terms:
            custom_terms.insert(0, contact_name.strip())
        for dg_kwargs in (
            {"keyterm": custom_terms, "filler_words": True, "endpointing_ms": 25, "no_delay": True},
            {"keyterm": custom_terms, "filler_words": True},
            {"keyterm": custom_terms},
            {"keyterms": custom_terms},
            {},
        ):
            try:
                primary_stt = deepgram.STT(
                    model="nova-3",
                    language=stt_lang,
                    api_key=deepgram_api_key,
                    **dg_kwargs,
                )
                break
            except TypeError:
                continue

    if cartesia_api_key and hasattr(cartesia, "STT"):
        try:
            backup_stt = cartesia.STT(
                model="ink-whisper",
                language=stt_lang,
                api_key=cartesia_api_key,
            )
        except Exception as exc:
            logger.debug("Cartesia backup STT note: %s", exc)

    if primary_stt is not None and backup_stt is not None and hasattr(lk_stt, "FallbackAdapter"):
        for kwarg_name in ("stt", "stts"):
            try:
                adapter = lk_stt.FallbackAdapter(**{kwarg_name: [primary_stt, backup_stt]})
                logger.info(
                    "🎧 Production Dual STT active: Primary='deepgram/nova-3' (Direct BYOK) | Fallback='cartesia/ink-whisper' (Direct BYOK)"
                )
                return adapter
            except TypeError:
                continue
            except Exception as exc:
                logger.debug("STT FallbackAdapter note: %s", exc)
                break

    if primary_stt is not None:
        logger.info("🎧 Direct Deepgram STT (BYOK) active: model='nova-3' | language='%s'", stt_lang)
        return primary_stt

    if backup_stt is not None:
        logger.info("🎧 Direct Cartesia STT (BYOK) active: model='ink-whisper' | language='%s'", stt_lang)
        return backup_stt

    logger.info("🎧 LiveKit Inference STT active: model='deepgram/nova-3' | language='%s'", stt_lang)
    return InferenceSTT(model="deepgram/nova-3", language=stt_lang)


def _build_tts(voice_id: str | None, language: str) -> lk_tts.TTS:
    """Construye el motor TTS priorizando BYOK directo (Deepgram Aura-2 para Celeste = 0 créditos Cartesia, o Cartesia sonic-3/sonic-latest)."""
    deepgram_api_key = os.environ.get("DEEPGRAM_API_KEY", "").strip()
    cartesia_api_key = os.environ.get("CARTESIA_API_KEY", "").strip()

    model = voice_id if voice_id else DEFAULT_TTS_MODEL
    voice: str | None = None
    if voice_id and ":" in voice_id:
        idx = voice_id.rfind(":")
        voice = voice_id[idx + 1:]
        model = voice_id[:idx]

    # 1. Si la voz seleccionada es de Deepgram (ej. Celeste 'deepgram/aura-2:celeste-es'), usar Deepgram Direct Plugin (0 créditos de Cartesia)
    if model.startswith("deepgram/") and deepgram_api_key:
        lang_suffix = "es" if language == "es" else "en"
        clean_voice = re.sub(r"-(es|en)$", "", (voice or "celeste")).strip()
        dg_tts_model = f"aura-2-{clean_voice}-{lang_suffix}"
        logger.info(
            "🎙️ Deepgram Direct TTS (BYOK - 0 Cartesia credits) active: model='%s' | language='%s'",
            dg_tts_model,
            language,
        )
        return deepgram.TTS(model=dg_tts_model, api_key=deepgram_api_key)

    # 2. Si la voz seleccionada es de Cartesia (ej. 'cartesia/sonic-3:...' en inglés o voz clonada)
    if model.startswith("cartesia/") and cartesia_api_key:
        raw_cartesia_model = model.split("/", 1)[1] if "/" in model else model
        cartesia_model = (
            "sonic-latest"
            if raw_cartesia_model in ("sonic-2", "sonic", "sonic-multilingual")
            else raw_cartesia_model
        )
        cartesia_kwargs: dict = {
            "model": cartesia_model,
            "language": language,
            "api_key": cartesia_api_key,
        }
        if voice:
            cartesia_kwargs["voice"] = voice

        primary_tts = cartesia.TTS(**cartesia_kwargs)

        # Si se pidió 'sonic-3', agregar respaldo automático a 'sonic-latest' y Deepgram TTS por si el endpoint o los créditos fallan
        fallback_list: list[lk_tts.TTS] = [primary_tts]
        if cartesia_model == "sonic-3":
            fallback_kwargs = dict(cartesia_kwargs, model="sonic-latest")
            fallback_list.append(cartesia.TTS(**fallback_kwargs))
        if deepgram_api_key:
            dg_backup_model = "aura-2-celeste-es" if language == "es" else "aura-2-thalia-en"
            fallback_list.append(deepgram.TTS(model=dg_backup_model, api_key=deepgram_api_key))

        if len(fallback_list) > 1 and hasattr(lk_tts, "FallbackAdapter"):
            for kwarg_name in ("tts", "ttss"):
                try:
                    adapter = lk_tts.FallbackAdapter(**{kwarg_name: fallback_list})
                    logger.info(
                        "🎙️ Cartesia Direct Plugin (BYOK) active: model='%s' | voice='%s' | language='%s' (with auto-fallback)",
                        cartesia_model,
                        voice,
                        language,
                    )
                    return adapter
                except TypeError:
                    continue

        logger.info(
            "🎙️ Cartesia Direct Plugin (BYOK) active: model='%s' | voice='%s' | language='%s'",
            cartesia_model,
            voice,
            language,
        )
        return primary_tts

    # 3. Respaldo general si no hay llave de Cartesia pero sí de Deepgram
    if deepgram_api_key:
        dg_default = "aura-2-celeste-es" if language == "es" else "aura-2-thalia-en"
        logger.info("🎙️ Deepgram Direct TTS Fallback active: model='%s'", dg_default)
        return deepgram.TTS(model=dg_default, api_key=deepgram_api_key)

    tts_kwargs: dict = {"model": model}
    if voice:
        tts_kwargs["voice"] = voice
    if not model.startswith("rime/"):
        tts_kwargs["language"] = language
    return InferenceTTS(**tts_kwargs)


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
    phone_number: str = (
        metadata.get("phone_number")
        or os.environ.get("DEFAULT_CLIENT_PHONE", "+584241226796")
    )
    language: str = metadata.get("language") or os.environ.get("AGENT_LANGUAGE", "es")
    default_instructions = (
        (
            "Representas a Zen Spa. El objetivo es invitar al cliente a aprovechar su cincuenta por ciento de descuento "
            "en paquetes de depilación láser definitiva y agendarle una sesión de valoración diagnóstica gratuita de quince minutos "
            "con nuestra especialista para evaluar su piel sin compromiso."
        )
        if language == "es"
        else (
            "You represent Zen Spa. Your goal is to invite the client to use their pre-approved fifty percent discount "
            "on laser hair removal packages and schedule a free fifteen-minute diagnostic skin assessment with our specialist "
            "with no obligation."
        )
    )
    agent_instructions: str = metadata.get("agent_instructions") or default_instructions

    use_cloned_voice = os.environ.get("USE_CLONED_VOICE", "0") == "1"
    default_test_voice = DEFAULT_ES_TEST_VOICE if language == "es" else DEFAULT_EN_TEST_VOICE
    voice_id: str | None = (
        metadata.get("voice_id")
        or (os.environ.get("VOICE_ID") if use_cloned_voice else None)
        or default_test_voice
    )
    voice_lower = (voice_id or "").lower()
    if "celeste" in voice_lower:
        default_agent_name = "Celeste"
    elif "f786b574-daa5-4673-aa0c-cbe3e8534c02" in voice_lower or "thalia" in voice_lower:
        default_agent_name = "Sophia" if language == "en" else "Daniela"
    elif any(k in voice_lower for k in ("5c5ad5e7-1020-476b-8b91-fdcbe9cc313c", "serena", "antonia", "aquila", "estrella", "diana", "carina", "selena")):
        default_agent_name = "Daniela"
    else:
        default_agent_name = "Franlex"

    agent_display_name: str = (
        metadata.get("agent_name")
        or (os.environ.get("AGENT_DISPLAY_NAME") if use_cloned_voice else None)
        or default_agent_name
    )
    contact_name: str = metadata.get("contact_name", "Franlex")
    default_context = (
        "Cliente frecuente y leal de Zen Spa. Tiene pre-aprobado un cincuenta por ciento de descuento en depilación láser definitiva y valoración inicial sin costo."
        if language == "es"
        else "Frequent and loyal client at Zen Spa. Pre-approved for a fifty percent discount on laser hair removal and a free initial diagnostic assessment."
    )
    personalization_context: str = metadata.get("personalization_context") or default_context
    custom_business_hours: str = (
        metadata.get("business_hours")
        or os.environ.get("DEFAULT_BUSINESS_HOURS", "")
    )

    temporal_context = _get_temporal_context(language, custom_business_hours)
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
            f"    Eres {agent_display_name}, especialista de atención al cliente en llamadas salientes. "
            "Asumes siempre que el cliente está ocupado haciendo otras cosas: tu ritmo es ágil, cercano, empático y ejecutivo, sin rodeos ni frases robóticas. JAMÁS uses términos en inglés como 'Senior Client Success Manager' al hablar en español.\n"
            "  </identity>\n\n"
            "  <execution_pipeline>\n"
            "    <phase index=\"1\" name=\"turno_1_presentacion_con_nombre_y_gancho_corto\">\n"
            f"      En cuanto el cliente confirme que es él después del 'Aló', salúdalo por su nombre en 2 palabras seguidas de PUNTO SEGUIDO, preséntate siempre con tu nombre ({agent_display_name}) y la empresa de donde llamas, menciona su beneficio principal según <business_instructions> y cierra con una pregunta corta de permiso (máximo 25 palabras en total; por ejemplo: 'Hola, {client_label}. Te habla {agent_display_name}, de Zen Spa; te llamo porque tienes aprobado un cincuenta por ciento de descuento en depilación láser por ser cliente frecuente. ¿Tienes treinta segundos?').\n"
            "    </phase>\n"
            "    <phase index=\"2\" name=\"turno_2_el_valor_y_validacion_de_interes\">\n"
            "      Cuando el cliente diga 'Sí, dime', 'Sí, ok' o pregunte '¿De qué trata?', NO pidas fechas todavía. Abre con una expresión humana natural ('Ah, mira.', 'Mmm, te cuento.') y explica brevemente en qué consiste el beneficio o valoración sin compromiso según <business_instructions> (máximo 20 palabras), preguntando si le gustaría aprovecharlo.\n"
            "    </phase>\n"
            "    <phase index=\"3\" name=\"turno_3_objeciones_humanas_y_agenda_con_muletillas\">\n"
            "      - PASO 3.1 (Manejo Humano de Dudas/Objeciones + Preguntar Día): Si el cliente acepta sin dudas ('Sí, dale', 'Me interesa'), pregúntale qué día de esta semana o de la próxima le queda bien. Si expresa una duda o temor (ej. pregunta si duele, si es gratis o cuánto dura), responde primero con calidez y naturalidad humana ('Jaja, no tranquilo.', 'Ah, no te preocupes.', 'Mmm, sí claro, es totalmente sin costo.') y en esa misma intervención pregúntale qué día de esta semana o la próxima le queda bien. PROHIBIDO pegar 'Excelente.' en medio de las dos oraciones.\n"
            "      - PASO 3.2 (Dar Disponibilidad con Muletillas Humanas en 2 Opciones): Cuando el cliente mencione un día o pida buscar disponibilidad, abre con una muletilla humana pensativa ('Mmm, veamos.', 'Eh, mira.', 'Mmm, déjame ver.') y ofrece SOLO DOS opciones primero (una en la mañana y una en la tarde) usando <temporal_awareness> para esta semana o la próxima, o check_availability si es de dos o más semanas en el futuro.\n"
            "      - PASO 3.3 (Otros Horarios, Horas No Disponibles y Selección Directa):\n"
            "        * Si pregunta qué otros horarios tienes en la mañana o en la tarde, abre con una muletilla natural ('Mmm, mira.', 'Eh, en la mañana...') y menciona de una vez las opciones restantes de ese bloque según <temporal_awareness>.\n"
            "        * Si pide una hora que NO está entre los cupos libres de <temporal_awareness>, explícale con naturalidad el motivo real según el horario del negocio (si está fuera del horario de atención, coméntaselo amablemente; si está dentro del horario pero no figura libre, dile que ese turno ya está ocupado) y ofrécele las opciones disponibles más cercanas.\n"
            "        * Si el cliente elige cualquiera de los horarios que TÚ le acabas de ofrecer, pasa DIRECTO al PASO 4.1 sin volver a preguntar '¿te lo reservo?'.\n"
            "    </phase>\n"
            "    <phase index=\"4\" name=\"confirmacion_natural_paso_a_paso_whatsapp_y_correo\">\n"
            "      PROHIBIDO pedir WhatsApp y correo en la misma pregunta y PROHIBIDO colgar sin verificar los datos. Haz una sola pregunta corta por turno y varía tus frases para no sonar repetitivo:\n"
            "      - PASO 4.1 (Confirmar Fecha/Hora sin Año y Preguntar WhatsApp): Apenas quede definida la hora, confirma de forma ágil el día, número de día y mes (SIN EL AÑO) y la hora, comenta que envías la confirmación por WhatsApp y correo, y pregunta si usan ese mismo número para su WhatsApp.\n"
            "      - PASO 4.2 (Rama Sí / Rama No y Verificación en Bucle del WhatsApp con Pausas):\n"
            "        * Si confirma que usa ese mismo número: pídele su correo electrónico.\n"
            "        * Si prefiere otro número: pregúntale a qué número de WhatsApp se la envías.\n"
            "        * VERIFICACIÓN OBLIGATORIA DEL TELÉFONO (Con comas obligatorias entre bloques): Cuando te dicte el número, repítelo con naturalidad incluyendo el 'cero' inicial y SEPARANDO SIEMPRE CON COMAS los bloques de dígitos para hacer pausas naturales (ej. 'Ah, perfecto. Te repito el número: cero cuatro veinticuatro, uno dos dos, seis siete nueve seis, ¿está bien así?'). PROHIBIDO decir 'mil', 'ciento' o 'cientos'. Si corrige algún dígito, ajústalo y vuelve a verificar hasta que confirme que SÍ. Recién ahí pídele el correo.\n"
            "      - PASO 4.3 (Verificación obligatoria del Correo con frase distinta al teléfono): Cuando te dicte su correo, NO invoques book_appointment todavía. Usa una frase distinta a la que usaste en el teléfono para repetirle el correo en palabras fonéticas ('punto' y 'arroba') y confirmar si está correcto.\n"
            "    </phase>\n"
            "    <phase index=\"5\" name=\"registro_final_con_fecha_hora_sin_ano_y_despedida_ejecutiva\">\n"
            f"      - ÚNICAMENTE cuando el cliente confirme que el correo repetido SÍ está correcto, invoca OBLIGATORIAMENTE book_appointment (pasando appointment_date en formato ISO YYYY-MM-DD y appointment_time en formato 24h HH:MM) y despídete de forma ejecutiva mencionando SIEMPRE el día, fecha (SIN EL AÑO) y hora exactos en letras, avisando que en segundos le llega por WhatsApp y correo, y deseándole que pase {current_farewell}.\n"
            f"      - Si en cualquier momento el cliente rechaza la llamada ('no me interesa', 'estoy ocupado'), despídete deseándole {current_farewell} e invoca end_call.\n"
            "    </phase>\n"
            "  </execution_pipeline>\n\n"
            "  <hard_constraints>\n"
            "    - MICRO-APERTURAS HUMANAS Y VARIADAS (MÁXIMO 2 PALABRAS ANTES DEL PUNTO): Inicia cada turno con 1 o 2 palabras seguidas de PUNTO SEGUIDO. Alterna expresiones humanas naturales ('Mmm, mira.', 'Eh, veamos.', 'Ah, claro.', 'Mmm, déjame ver.', 'Jaja, tranquilo.', 'Listo.') en lugar de repetir 'Claro.' o 'Perfecto.' en todos los turnos.\n"
            "    - PROHIBIDO MENCIONAR EL AÑO AL HABLAR: JAMÁS digas el año ('dos mil veintiséis' ni '2026') en tu respuesta de voz; di únicamente el día de la semana, el número y el mes (ej. 'jueves tres de diciembre').\n"
            "    - ECONOMÍA VERBAL (CLIENTE OCUPADO): Ninguna respuesta debe superar las 26 palabras. Ve directo al grano y haz solo una pregunta por turno.\n"
            "    - FONÉTICA ESTRICTA Y COMAS EN TELÉFONOS: Un solo párrafo continuo sin saltos de línea. PROHIBIDO usar signos de exclamación (¡ !). Escribe TODOS los números, fechas, porcentajes y horas en letras minúsculas. Al repetir teléfonos, separa SIEMPRE los bloques con comas ('cero cuatro veinticuatro, uno dos dos, seis siete nueve seis') y JAMÁS pronuncies 'mil', 'ciento', 'cientos' ni 'millones'.\n"
            "    - AISLAMIENTO DE HERRAMIENTAS: JAMÁS escribas ni pronuncies nombres de funciones o código en tu respuesta.\n"
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
            f"    You are {agent_display_name}, a client success specialist making an outbound call. "
            "Assume the prospect is busy: your style is brisk, warm, empathetic, and executive, without robotic phrasing. Adapt your words naturally rather than reciting memorized templates.\n"
            "  </identity>\n\n"
            "  <execution_pipeline>\n"
            "    <phase index=\"1\" name=\"turn_1_name_intro_and_short_hook\">\n"
            f"      As soon as the prospect confirms it is them after 'Hello?', greet them by name in 2 words followed by a PERIOD, introduce yourself by name ({agent_display_name}) and the company you are calling from, state their main benefit from <business_instructions>, and end with a quick permission check (max 25 words total; e.g., 'Hi, {client_label}. This is {agent_display_name} from Zen Spa; I am calling because you have a pre-approved fifty percent discount on laser hair removal as a frequent client. Do you have thirty seconds?').\n"
            "    </phase>\n"
            "    <phase index=\"2\" name=\"turn_2_value_and_interest_check\">\n"
            "      When the prospect says 'Sure / Go ahead' or asks 'What is this about?', DO NOT ask for dates yet. Open with a natural human filler ('Oh, look.', 'Mmm, so basically.') and briefly explain the no-obligation benefit or assessment from <business_instructions> (max 20 words), asking if they would like to take advantage of it.\n"
            "    </phase>\n"
            "    <phase index=\"3\" name=\"turn_3_human_objections_and_smart_scheduling\">\n"
            "      - STEP 3.1 (Human Objection Handling + Ask Day): If they accept without concerns ('Sure', 'Sounds good'), ask what day this week or next week works best. If they express a concern (e.g., whether it hurts, if it is free, or how long it takes), respond first with warm human reassurance ('Haha, don't worry.', 'Oh, no worries at all.', 'Mmm, yes, it is completely free.') and smoothly ask what day this week or next week works best. NEVER insert a robotic 'Excellent.' between the two sentences.\n"
            "      - STEP 3.2 (Offer 2 Options with Human Fillers): When they mention a day or ask for availability, open with a thoughtful human filler ('Mmm, let's see.', 'Uh, look.', 'Mmm, let me check.') and offer ONLY TWO options first (one morning, one afternoon) using <temporal_awareness> for this week or next week, or calling check_availability for dates two or more weeks ahead.\n"
            "      - STEP 3.3 (Other Slots, Unavailable Times & Direct Selection):\n"
            "        * If asked for other morning or afternoon times, open with a natural filler ('Mmm, look.', 'Uh, in the morning...') and offer the remaining available slots for that block from <temporal_awareness> at once.\n"
            "        * If they ask for a time NOT in the available slots of <temporal_awareness>, naturally explain why based on the schedule (if it is outside business operating hours, mention that warmly; if it is within hours but not listed as open, let them know that slot is already booked) and offer the closest available times.\n"
            "        * If they pick an option you JUST offered, go straight to STEP 4.1 without asking 'Should I book it?'.\n"
            "    </phase>\n"
            "    <phase index=\"4\" name=\"natural_step_by_step_whatsapp_and_email_verification\">\n"
            "      NEVER ask for WhatsApp and email in the same question, and NEVER hang up without verifying their data. Ask one short question per turn and vary your phrasing naturally:\n"
            "      - STEP 4.1 (Confirm Full Date/Time without Year & Ask WhatsApp): Once the time is chosen, state the full day of the week, month, ordinal day (WITHOUT THE YEAR), and time, mention the confirmation goes out via WhatsApp and email, and ask if you should use this same number for WhatsApp.\n"
            "      - STEP 4.2 (WhatsApp Yes / No Branch & Natural Phone Read-Back Loop with Commas):\n"
            "        * If YES: ask what email address you should send it to.\n"
            "        * If NO: ask what phone number is best for WhatsApp.\n"
            "        * MANDATORY PHONE VERIFICATION LOOP (With mandatory commas between digit groups): When they dictate the number, read it back naturally separating digit groups with COMMAS for natural pauses (NEVER say 'thousand' or 'hundred', e.g., 'Ah, got it. Let me read that back: zero four two four, one two two, six seven nine six, does that sound right?'). If they correct any digit, fix it and verify again until confirmed YES, then ask for their email.\n"
            "      - STEP 4.3 (Mandatory Email Read-Back Loop with varied phrasing): When they give their email, DO NOT invoke book_appointment yet. Use a different natural phrasing than the phone check to read back the email in spoken words ('dot' and 'at') and ask if it is correct until confirmed YES.\n"
            "    </phase>\n"
            "    <phase index=\"5\" name=\"final_booking_with_full_date_and_time_without_year\">\n"
            f"      - ONLY after the prospect confirms that the repeated email is YES correct, MANDATORILY invoke book_appointment (passing appointment_date in ISO YYYY-MM-DD format and appointment_time in 24h HH:MM format) and say a brisk executive goodbye stating the exact day, date (WITHOUT THE YEAR), and time in words, letting them know the confirmation will arrive via WhatsApp and email in seconds, and wishing them {current_farewell}.\n"
            f"      - If they reject the call, wish them {current_farewell} and invoke end_call.\n"
            "    </phase>\n"
            "  </execution_pipeline>\n\n"
            "  <hard_constraints>\n"
            "    - HUMAN VARIED MICRO-OPENERS (MAX 2 WORDS BEFORE PERIOD): Start every turn with 1 or 2 natural words followed by a PERIOD. Alternate conversational human openers ('Mmm, look.', 'Uh, let's see.', 'Ah, gotcha.', 'Oh, no worries.', 'Haha, relax.', 'Got it.') instead of repeating 'Sure.' or 'Perfect.' on every turn.\n"
            "    - NEVER MENTION THE YEAR OUT LOUD: NEVER say the year ('2026' or 'two thousand twenty-six') in your spoken reply; state only the day of the week, month, and ordinal day (e.g., 'Thursday, December third').\n"
            "    - VERBAL ECONOMY (BUSY PROSPECT): Keep every reply under 26 words and ask only one question per turn.\n"
            "    - STRICT PHONETICS & COMMAS IN PHONE NUMBERS: Reply in a single continuous paragraph without line breaks. NEVER use exclamation marks (!). Spell out ALL numbers, phone digits, dates, percentages, and times in lowercase words. Always separate phone digit groups with commas ('zero four two four, one two two, six seven nine six') and NEVER read phone numbers as 'hundreds' or 'thousands'.\n"
            "    - TOOL ISOLATION: NEVER write or speak function names or code in your text response.\n"
            "  </hard_constraints>\n"
            "</agent_system_manifest>"
        )

    logger.info(
        "Starting call as '%s' (voice=%s, language=%s)",
        agent_display_name, voice_id or DEFAULT_TTS_MODEL, language,
    )

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    tts = _build_tts(voice_id, language)

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

    stt = _build_stt(language, contact_name)
    llm_engine = _build_llm()

    session: AgentSession
    try:
        session = AgentSession(
            min_endpointing_delay=0.22,
            max_endpointing_delay=0.40,
            preemptive_generation=True,
            min_interruption_words=2,
            min_interruption_duration=0.5,
        )
    except TypeError:
        session = AgentSession(
            min_endpointing_delay=0.22,
            max_endpointing_delay=0.40,
            preemptive_generation=True,
        )

    call_ending = False
    booking_state: dict = {
        "last_user_text": "",
        "needs_readback_confirmation": False,
        "extracted_phone_digits": "",
        "force_remove_k_in_franlex": False,
        "last_checked_iso_date": "",
    }

    @llm.function_tool(
        description=(
            "Calculates in Python the exact calendar date when the client asks for dates two or more weeks ahead (in 2 weeks=2, in 3 weeks=3, in 4 weeks=4). "
            "For today, tomorrow, this week, or next week, use <temporal_awareness> directly."
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
            iso_d = target_dt.strftime("%Y-%m-%d")
            booking_state["last_checked_iso_date"] = iso_d
            phonetic = _format_date_phonetic(target_dt, language)
            if language == "es":
                return (
                    f"Fecha exacta calculada para pasado mañana: {phonetic} (ISO para base de datos: {iso_d}). "
                    "INSTRUCCIÓN: Abre con una muletilla humana ('Mmm, mira.', 'Eh, veamos.'), NO menciones el año al hablar y ofrece SOLO DOS opciones según <temporal_awareness>."
                )
            return (
                f"Exact date for day after tomorrow: {phonetic} (Database ISO: {iso_d}). "
                "INSTRUCTION: Open with a human filler ('Mmm, look.', 'Uh, let's see.'), DO NOT mention the year out loud, and offer ONLY TWO options first based on <temporal_awareness>."
            )

        if "mañana" in clean_day or "tomorrow" in clean_day:
            target_dt = now + timedelta(days=1)
            iso_d = target_dt.strftime("%Y-%m-%d")
            booking_state["last_checked_iso_date"] = iso_d
            phonetic = _format_date_phonetic(target_dt, language)
            if language == "es":
                return (
                    f"Fecha exacta calculada para mañana: {phonetic} (ISO para base de datos: {iso_d}). "
                    "INSTRUCCIÓN: Abre con una muletilla humana ('Mmm, mira.', 'Eh, veamos.'), NO menciones el año al hablar y ofrece SOLO DOS opciones según <temporal_awareness>."
                )
            return (
                f"Exact date for tomorrow: {phonetic} (Database ISO: {iso_d}). "
                "INSTRUCTION: Open with a human filler ('Mmm, look.', 'Uh, let's see.'), DO NOT mention the year out loud, and offer ONLY TWO options first based on <temporal_awareness>."
            )

        if "hoy" in clean_day or "today" in clean_day or "para hoy" in last_u_day or "es hoy" in last_u_day:
            iso_d = now.strftime("%Y-%m-%d")
            booking_state["last_checked_iso_date"] = iso_d
            phonetic = _format_date_phonetic(now, language)
            logger.info("📅 [CALENDAR TOOL] day='%s' (HOY/TODAY) -> %s (%s)", day_of_week, phonetic, iso_d)
            if language == "es":
                return (
                    f"Fecha exacta calculada para HOY: {phonetic} (ISO para base de datos: {iso_d}). "
                    "INSTRUCCIÓN: Abre con una muletilla humana, NO menciones el año al hablar y ofrece SOLO DOS opciones disponibles hoy según <temporal_awareness>."
                )
            return (
                f"Exact date for TODAY: {phonetic} (Database ISO: {iso_d}). "
                "INSTRUCTION: Open with a human filler, DO NOT mention the year out loud, and offer ONLY TWO options available today based on <temporal_awareness>."
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
        req_iso = requested_dt.strftime("%Y-%m-%d")
        booking_state["last_checked_iso_date"] = req_iso

        req_phonetic = _format_date_phonetic(requested_dt, language)
        this_wk_phonetic = _format_date_phonetic(this_week_dt, language)
        next_wk_phonetic = _format_date_phonetic(next_week_dt, language)
        two_wk_phonetic = _format_date_phonetic(two_weeks_dt, language)

        logger.info(
            "📅 [CALENDAR TOOL] day='%s' weeks_ahead=%d -> %s (%s)",
            day_of_week,
            effective_weeks,
            req_phonetic,
            req_iso,
        )

        if language == "es":
            return (
                f"RESULTADO EXACTO DE CALENDARIO PYTHON:\n"
                f"- Fecha solicitada ({effective_weeks} semanas adelante): {req_phonetic} (ISO para book_appointment: {req_iso}).\n"
                f"- Desglose de seguridad: En esta semana actual es {this_wk_phonetic}; "
                f"en la próxima semana (semana entrante) es {next_wk_phonetic}; "
                f"en dos semanas es {two_wk_phonetic}.\n"
                f"- INSTRUCCIÓN DE RESPUESTA: Abre con una expresión humana natural ('Mmm, veamos.', 'Eh, mira.'), NO digas el año al hablar y ofrece primero SOLO DOS opciones (una en la mañana y una en la tarde) según <temporal_awareness>."
            )
        return (
            f"EXACT PYTHON CALENDAR RESULT:\n"
            f"- Requested date ({effective_weeks} weeks ahead): {req_phonetic} (ISO for book_appointment: {req_iso}).\n"
            f"- Reference breakdown: This week is {this_wk_phonetic}; next week is {next_wk_phonetic}; in two weeks is {two_wk_phonetic}.\n"
            f"- RESPONSE INSTRUCTION: Open with a natural human filler ('Mmm, let's see.', 'Uh, look.'), DO NOT say the year out loud, and offer ONLY TWO options first (one morning, one afternoon) based on <temporal_awareness>."
        )

    @llm.function_tool(
        description=(
            "Formally books the appointment in Supabase ONLY AFTER reading back both the WhatsApp number (if a different one was given) and the email address "
            "and receiving the client's YES confirmation. Pass appointment_date in ISO 'YYYY-MM-DD' format and appointment_time in 24h 'HH:MM' format."
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

        if booking_state.get("force_remove_k_in_franlex"):
            email = re.sub(r"(?i)franklex", "franlex", email)

        # Candado determinista bilingüe (ES/EN): si el cliente acaba de dictar o corregir teléfono/correo en este mismo turno, obligar a repetirlo y esperar confirmación
        just_dictated_or_corrected_data = (
            booking_state["needs_readback_confirmation"]
            or "arroba" in last_u
            or "@" in last_u
            or "gmail" in last_u
            or "hotmail" in last_u
            or "outlook" in last_u
            or "yahoo" in last_u
            or "icloud" in last_u
            or "dot com" in last_u
            or "incorrecto" in last_u
            or "equivocado" in last_u
            or "wrong" in last_u
            or "incorrect" in last_u
            or "otro número" in last_u
            or "otro numero" in last_u
            or "diferente" in last_u
            or "another number" in last_u
            or "different number" in last_u
        )
        if just_dictated_or_corrected_data:
            logger.info(
                "🛡️ [GUARDRAIL] Blocked premature book_appointment — forcing agent to read back and confirm phone/email ('%s') first.",
                email,
            )
            if any(k in last_u for k in ("otro número", "otro numero", "diferente", "another number", "different number")):
                if language == "es":
                    return (
                        "ALTO - NO CIERRES LA LLAMADA TODAVÍA: El cliente acaba de indicar que usará otro número telefónico. "
                        "Pídele primero el número de WhatsApp, repíteselo con comas entre bloques para confirmar si está correcto, y luego verifica su correo electrónico."
                    )
                return (
                    "STOP - DO NOT END THE CALL YET: The prospect just indicated they want to use another phone number. "
                    "Ask for their WhatsApp number first, read it back with commas between groups to confirm it is correct, and then verify their email address."
                )
            if language == "es":
                return (
                    f"ALTO - NO CIERRES LA LLAMADA TODAVÍA: El cliente acaba de dictar o corregir un dato de contacto ('{last_u}'). "
                    f"Según el protocolo obligatorio, primero debes repetirle el dato ('{email}') en palabras y preguntarle si está correcto. "
                    "Solo cuando el cliente te responda que sí en el siguiente turno, vuelve a invocar book_appointment."
                )
            return (
                f"STOP - DO NOT END THE CALL YET: The prospect just dictated or corrected contact info ('{last_u}'). "
                f"You must first read back the info ('{email}') in spoken words and ask if it is correct. "
                "Only after they confirm yes on the next turn, invoke book_appointment again."
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

        iso_date, iso_time, iso_full_at, spoken_date, spoken_time = _normalize_booking_datetime(
            appointment_date,
            appointment_time,
            language,
            booking_state.get("last_checked_iso_date", ""),
        )

        # Payload estructurado 100% compatible con tipos DATE, TIME y TIMESTAMPTZ de Supabase (PostgreSQL)
        booking_payload = {
            "client_name": contact_name or ("Cliente" if language == "es" else "Prospect"),
            "appointment_date": iso_date,
            "appointment_time": iso_time,
            "appointment_at_iso": iso_full_at,
            "appointment_display": (
                f"{spoken_date} a las {spoken_time}"
                if language == "es"
                else f"{spoken_date} at {spoken_time}"
            ),
            "whatsapp_number": resolved_phone,
            "email": email.strip().lower(),
            "language": language,
            "booked_at_iso": datetime.now(ZoneInfo("America/Caracas")).isoformat(),
        }
        logger.info("✅ [BOOKING CONFIRMED - SUPABASE READY] %s", json.dumps(booking_payload, ensure_ascii=False))

        if not call_ending:
            call_ending = True
            logger.info("📴 Booking completed — waiting for full confirmation & goodbye audio before hanging up…")
            asyncio.create_task(_terminate_call_session(ctx, session, initial_wait=2.5, max_wait=18.0))

        _, _, dynamic_farewell = _get_time_of_day_phrases(language)
        if language == "es":
            return (
                f"Cita registrada en formato Supabase ({iso_date} {iso_time}). "
                f"Ahora despídete de forma ágil y directa mencionando OBLIGATORIAMENTE el día, fecha y hora en letras EXACTAMENTE así (SIN DECIR EL AÑO): "
                f"' el {spoken_date} a las {spoken_time}', avisando que en segundos le llega la confirmación por WhatsApp y correo, "
                f"y deseándole que pase {dynamic_farewell}."
            )
        return (
            f"Appointment recorded in Supabase format ({iso_date} {iso_time}). "
            f"Now say a concise, warm goodbye stating the exact day, date, and time in words EXACTLY as: "
            f"'{spoken_date} at {spoken_time}' (WITHOUT THE YEAR), letting them know the confirmation will arrive via WhatsApp and email in seconds, "
            f"and wishing them {dynamic_farewell}."
        )

    @llm.function_tool(
        description=(
            "Hangs up and ends the phone call. Invoke it when the client rejects the service/call or after saying goodbye."
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
            text = _sanitize_stt_text(text, contact_name)
            if booking_state.get("force_remove_k_in_franlex"):
                text = re.sub(r"(?i)\bfranklex\b", "franlex", text)
            lower_u = text.lower()
            if (
                "sin la k" in lower_u
                or "pusiste una k" in lower_u
                or "agregaste una k" in lower_u
                or "letra k" in lower_u
                or "without the k" in lower_u
                or "without k" in lower_u
                or "no k" in lower_u
                or "letter k" in lower_u
            ):
                booking_state["force_remove_k_in_franlex"] = True
                text = re.sub(r"(?i)\bfranklex\b", "franlex", text)
                lower_u = text.lower()
            booking_state["last_user_text"] = text
            parsed_digits = _extract_phone_digits_from_text(text)
            if parsed_digits:
                booking_state["extracted_phone_digits"] = parsed_digits
                booking_state["needs_readback_confirmation"] = True
                logger.info("📞 [PHONE PARSER] Captured exact E.164 phone from user speech: %s", parsed_digits)
            if (
                "arroba" in lower_u
                or "@" in lower_u
                or "gmail" in lower_u
                or "hotmail" in lower_u
                or "outlook" in lower_u
                or "yahoo" in lower_u
                or "icloud" in lower_u
                or "dot com" in lower_u
                or "incorrecto" in lower_u
                or "equivocado" in lower_u
                or "wrong" in lower_u
                or "incorrect" in lower_u
                or "agregaste" in lower_u
                or "sin la" in lower_u
                or "without the" in lower_u
                or "otro número" in lower_u
                or "otro numero" in lower_u
                or "another number" in lower_u
                or "different number" in lower_u
            ):
                booking_state["needs_readback_confirmation"] = True
            print(f"\n👤 [{contact_name or 'Usuario'}]: {text}", flush=True)
        elif role == "assistant":
            clean_text = text.replace("functions.end_call", "").replace("end_call()", "").strip()
            booking_state["needs_readback_confirmation"] = False
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

    # Watchdog de seguridad financiera: corta la llamada si supera max_call_duration (por defecto 300s = 5 min)
    max_duration = int(metadata.get("max_call_duration", os.environ.get("MAX_CALL_DURATION_SECONDS", "300")))
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
            greeting = f"Say only: 'Hello... hi, am I speaking with {contact_name}?' Wait for their response before introducing yourself."
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