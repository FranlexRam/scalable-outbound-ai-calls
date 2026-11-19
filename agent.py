import asyncio
from datetime import datetime, timedelta
import json
import logging
import os
import re
from urllib.parse import quote
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import httpx
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

try:
    from google.auth.transport.requests import Request as GoogleAuthRequest
    from google.oauth2 import service_account as google_service_account
except ImportError:
    GoogleAuthRequest = None
    google_service_account = None

# Pre-importar módulos internos de esquemas de herramientas para evitar bloqueos de 200ms en el Event Loop
try:
    import livekit.agents.llm.async_toolset  # noqa: F401
    from livekit.agents.llm import utils as lk_llm_utils  # noqa: F401
except ImportError:
    lk_llm_utils = None

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
DEFAULT_TENANT_ID = os.environ.get("SPA_TENANT_ID", "zen_spa")
DEFAULT_BUSINESS_NAME = os.environ.get("SPA_BUSINESS_NAME", "Zen Spa")
DEFAULT_SERVICE_NAME = os.environ.get("SPA_SERVICE_NAME", "Valoración Diagnóstica Láser (50% OFF)")
DEFAULT_BUSINESS_ADDRESS = os.environ.get(
    "SPA_ADDRESS",
    "Av. Principal, Centro Comercial Torre Zen, Piso 2, Local 24",
)
DEFAULT_FROM_EMAIL = os.environ.get("RESEND_FROM_EMAIL", "Zen Spa <contact@plexuscode.com>")
DEFAULT_REPLY_TO_EMAIL = os.environ.get("SPA_REPLY_TO_EMAIL", "")
DEFAULT_ADMIN_EMAIL = os.environ.get("SPA_ADMIN_EMAIL", "")
DEFAULT_GOOGLE_CALENDAR_ID = os.environ.get("GOOGLE_CALENDAR_ID", "").strip().strip("'\"")
DEFAULT_GOOGLE_SA_FILE = os.environ.get("GOOGLE_SERVICE_ACCOUNT_FILE", "google_service_account.json").strip().strip("'\"")
# Horario corrido de 9:00 AM a 5:00 PM (incluyendo 12:00 PM mediodía, 1:00 PM y 2:00 PM)
DEFAULT_MORNING_HOURS = (9, 10, 11)
DEFAULT_AFTERNOON_HOURS = (12, 13, 14, 15, 16, 17)
DEFAULT_TTS_MODEL = "cartesia/sonic-2"
DEFAULT_ES_TEST_VOICE = "deepgram/aura-2:celeste-es"
DEFAULT_EN_TEST_VOICE = "deepgram/aura-2:thalia-en"
DEFAULT_LIVEKIT_LLM_MODEL = os.environ.get("LIVEKIT_LLM_MODEL", "openai/gpt-4o-mini")
DEFAULT_OPENROUTER_LLM_MODEL = os.environ.get("OPENROUTER_LLM_MODEL", "google/gemini-2.5-flash")
SECONDARY_OPENROUTER_LLM_MODEL = os.environ.get("SECONDARY_OPENROUTER_LLM_MODEL", "anthropic/claude-haiku-5.5")
QUOTA_STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".lk_quota_state.json")

_SHARED_SUPABASE_CLIENT: httpx.AsyncClient | None = None
_GOOGLE_CAL_CREDS = None


def _get_supabase_http_client() -> httpx.AsyncClient:
    """Devuelve un cliente HTTP persistente (Keep-Alive TLS) para eliminar ~600ms de handshake en cada operación con Supabase, Resend y Google Calendar."""
    global _SHARED_SUPABASE_CLIENT
    if _SHARED_SUPABASE_CLIENT is None or _SHARED_SUPABASE_CLIENT.is_closed:
        _SHARED_SUPABASE_CLIENT = httpx.AsyncClient(
            timeout=5.0,
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20, keepalive_expiry=120.0),
        )
    return _SHARED_SUPABASE_CLIENT


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
    "mediodia": 12, "mediodía": 12, "noon": 12, "midday": 12,
}
HOURS_ES_WORDS = {
    1: "una", 2: "dos", 3: "tres", 4: "cuatro", 5: "cinco", 6: "seis",
    7: "siete", 8: "ocho", 9: "nueve", 10: "diez", 11: "once", 12: "doce",
}
HOURS_EN_WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
    7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve",
}


def _format_hour_spoken(hour_24: int, language: str) -> str:
    """Formatea cualquier hora (9..17) en palabras naturales incluyendo 'doce del mediodía' / 'twelve noon'."""
    if language == "es":
        if hour_24 == 12:
            return "doce del mediodía"
        if hour_24 < 12:
            return f"{HOURS_ES_WORDS[hour_24]} de la mañana"
        return f"{HOURS_ES_WORDS[hour_24 - 12]} de la tarde"
    if hour_24 == 12:
        return "twelve noon"
    if hour_24 < 12:
        return f"{HOURS_EN_WORDS[hour_24]} in the morning"
    return f"{HOURS_EN_WORDS[hour_24 - 12]} in the afternoon"


def _format_written_datetime(iso_date: str, iso_time: str, language: str) -> str:
    """Formatea fecha y hora en estilo ejecutivo limpio para correos y paneles (ej. 'Viernes, 27 de noviembre de 2026 · 03:00 PM')."""
    try:
        dt = datetime.strptime(f"{iso_date} {iso_time[:5]}", "%Y-%m-%d %H:%M")
        hour_24 = dt.hour
        minute_val = dt.minute
        period_ampm = "AM" if hour_24 < 12 else "PM"
        hour_12 = hour_24 % 12 or 12
        time_12h_str = f"{hour_12:02d}:{minute_val:02d} {period_ampm}"
        if language == "es":
            day_name = DAYS_ES[dt.weekday()].capitalize()
            month_name = MONTHS_ES[dt.month - 1]
            return f"{day_name}, {dt.day} de {month_name} de {dt.year} · {time_12h_str}"
        day_name_en = DAYS_EN[dt.weekday()]
        month_name_en = dt.strftime("%B")
        return f"{day_name_en}, {month_name_en} {dt.day}, {dt.year} · {time_12h_str}"
    except Exception:
        return f"{iso_date} · {iso_time[:5]}"


def _get_supabase_credentials() -> tuple[str, str]:
    """Obtiene y sanea URL y API Key de Supabase desde las variables de entorno."""
    sb_url = (
        os.environ.get("SUPABASE_URL")
        or os.environ.get("NEXT_PUBLIC_SUPABASE_URL")
        or ""
    ).strip().strip("'\"")
    sb_key = (
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        or os.environ.get("SUPABASE_KEY")
        or os.environ.get("SUPABASE_ANON_KEY")
        or os.environ.get("NEXT_PUBLIC_SUPABASE_ANON_KEY")
        or ""
    ).strip().strip("'\"")

    if sb_url and not sb_url.startswith("http"):
        sb_url = f"https://{sb_url}.supabase.co" if "." not in sb_url else f"https://{sb_url}"
    sb_url = sb_url.rstrip("/")

    if "tu-proyecto" in sb_url or "your-project" in sb_url:
        logger.warning(
            "⚠️ [SUPABASE CONFIG] SUPABASE_URL contiene el placeholder '%s'. Actualízalo en tu .env con la URL real de tu proyecto.",
            sb_url,
        )
        return "", ""
    return sb_url, sb_key


async def _get_google_calendar_access_token() -> str:
    """Obtiene y cachea en RAM un token OAuth2 de la Service Account sin bloquear el Event Loop de voz."""
    global _GOOGLE_CAL_CREDS
    if google_service_account is None or GoogleAuthRequest is None:
        logger.warning("⚠️ [GCAL SKIPPED] google-auth is not installed in this venv.")
        return ""

    sa_file = os.environ.get("GOOGLE_SERVICE_ACCOUNT_FILE", DEFAULT_GOOGLE_SA_FILE).strip().strip("'\"")
    if not os.path.isabs(sa_file):
        sa_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), sa_file)
    if not os.path.exists(sa_file):
        logger.warning("⚠️ [GCAL SKIPPED] Service Account file not found at '%s'.", sa_file)
        return ""

    try:
        if _GOOGLE_CAL_CREDS is None:
            _GOOGLE_CAL_CREDS = google_service_account.Credentials.from_service_account_file(
                sa_file,
                scopes=["https://www.googleapis.com/auth/calendar"],
            )
        if not _GOOGLE_CAL_CREDS.valid or not _GOOGLE_CAL_CREDS.token:
            await asyncio.to_thread(_GOOGLE_CAL_CREDS.refresh, GoogleAuthRequest())
        return _GOOGLE_CAL_CREDS.token or ""
    except Exception as exc:
        logger.error("❌ [GCAL AUTH ERROR] Failed to obtain access token: %s", exc)
        return ""


async def _fetch_supabase_booked_slots(
    tenant_id: str,
    start_iso_date: str,
    end_iso_date: str,
) -> dict[str, set[int]]:
    """Pre-carga usando el cliente HTTP Keep-Alive todos los horarios ya ocupados en Supabase para los próximos 14 días."""
    sb_url, sb_key = _get_supabase_credentials()
    if not (sb_url and sb_key):
        logger.warning("⚠️ [SUPABASE] Missing valid SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in .env — using default open schedule.")
        return {}

    endpoint = f"{sb_url}/rest/v1/spa_appointments"
    headers = {
        "apikey": sb_key,
        "Authorization": f"Bearer {sb_key}",
    }
    params = [
        ("tenant_id", f"eq.{tenant_id}"),
        ("appointment_date", f"gte.{start_iso_date}"),
        ("appointment_date", f"lte.{end_iso_date}"),
        ("status", "in.(scheduled,confirmed,rescheduled)"),
        ("select", "appointment_date,appointment_time"),
    ]
    booked_map: dict[str, set[int]] = {}
    try:
        client = _get_supabase_http_client()
        resp = await client.get(endpoint, headers=headers, params=params)
        if resp.status_code == 200:
            rows = resp.json()
            for row in rows:
                d_str = str(row.get("appointment_date", "")).strip()
                t_str = str(row.get("appointment_time", "")).strip()
                if d_str and t_str:
                    hour_val = int(t_str.split(":")[0])
                    booked_map.setdefault(d_str, set()).add(hour_val)
            logger.info(
                "🗄️ [SUPABASE PRE-FETCH] Loaded %d active bookings from '%s' for tenant='%s' (%s to %s).",
                len(rows),
                sb_url,
                tenant_id,
                start_iso_date,
                end_iso_date,
            )
        else:
            logger.warning("⚠️ [SUPABASE PRE-FETCH] HTTP %d from '%s': %s", resp.status_code, sb_url, resp.text[:200])
    except Exception as exc:
        logger.warning("⚠️ [SUPABASE PRE-FETCH] Could not reach '%s' (%s) — falling back to open schedule.", sb_url, exc)
    return booked_map


async def _fetch_google_calendar_booked_slots(
    calendar_id: str,
    start_iso_date: str,
    end_iso_date: str,
) -> dict[str, set[int]]:
    """Lee en tiempo real los eventos ocupados del Google Calendar del Spa para detectar bloqueos manuales."""
    cal_id = calendar_id.strip().strip("'\"")
    if not cal_id:
        return {}
    token = await _get_google_calendar_access_token()
    if not token:
        return {}

    tz = ZoneInfo("America/Caracas")
    time_min = f"{start_iso_date}T00:00:00-04:00"
    time_max = f"{end_iso_date}T23:59:59-04:00"
    endpoint = f"https://www.googleapis.com/calendar/v3/calendars/{quote(cal_id, safe='')}/events"
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "timeMin": time_min,
        "timeMax": time_max,
        "singleEvents": "true",
        "orderBy": "startTime",
    }

    booked_map: dict[str, set[int]] = {}
    try:
        client = _get_supabase_http_client()
        resp = await client.get(endpoint, headers=headers, params=params)
        if resp.status_code != 200:
            logger.warning("⚠️ [GCAL PRE-FETCH] HTTP %d: %s", resp.status_code, resp.text[:200])
            return {}

        items = resp.json().get("items", [])
        all_business_hours = set(DEFAULT_MORNING_HOURS + DEFAULT_AFTERNOON_HOURS)
        for ev in items:
            if ev.get("status") == "cancelled" or ev.get("transparency") == "transparent":
                continue
            start_obj = ev.get("start", {})
            end_obj = ev.get("end", {})

            if "dateTime" in start_obj:
                s_raw = str(start_obj["dateTime"]).replace("Z", "+00:00")
                e_raw = str(end_obj.get("dateTime", start_obj["dateTime"])).replace("Z", "+00:00")
                s_dt = datetime.fromisoformat(s_raw).astimezone(tz)
                e_dt = datetime.fromisoformat(e_raw).astimezone(tz)
                if e_dt <= s_dt:
                    e_dt = s_dt + timedelta(hours=1)
                cursor = s_dt.replace(minute=0, second=0, microsecond=0)
                while cursor < e_dt:
                    d_key = cursor.strftime("%Y-%m-%d")
                    if start_iso_date <= d_key <= end_iso_date and cursor.hour in all_business_hours:
                        booked_map.setdefault(d_key, set()).add(cursor.hour)
                    cursor += timedelta(hours=1)
            elif "date" in start_obj:
                s_date = datetime.strptime(start_obj["date"], "%Y-%m-%d").date()
                e_date = datetime.strptime(end_obj.get("date", start_obj["date"]), "%Y-%m-%d").date()
                if e_date <= s_date:
                    e_date = s_date + timedelta(days=1)
                d_cur = s_date
                while d_cur < e_date:
                    d_key = d_cur.strftime("%Y-%m-%d")
                    if start_iso_date <= d_key <= end_iso_date:
                        booked_map.setdefault(d_key, set()).update(all_business_hours)
                    d_cur += timedelta(days=1)

        logger.info(
            "📆 [GCAL PRE-FETCH] Loaded %d active events from Google Calendar (%s to %s).",
            len(items),
            start_iso_date,
            end_iso_date,
        )
    except Exception as exc:
        logger.warning("⚠️ [GCAL PRE-FETCH EXCEPTION] %s", exc)
    return booked_map


async def _fetch_combined_booked_slots(
    tenant_id: str,
    calendar_id: str,
    start_iso_date: str,
    end_iso_date: str,
) -> dict[str, set[int]]:
    """Ejecuta en paralelo la consulta de Supabase y Google Calendar y fusiona ambos mapas de horarios ocupados."""
    sb_map, gcal_map = await asyncio.gather(
        _fetch_supabase_booked_slots(tenant_id, start_iso_date, end_iso_date),
        _fetch_google_calendar_booked_slots(calendar_id, start_iso_date, end_iso_date),
    )
    merged: dict[str, set[int]] = {}
    for src in (sb_map, gcal_map):
        for d_key, hours_set in src.items():
            merged.setdefault(d_key, set()).update(hours_set)
    return merged


async def _fetch_booked_hours_for_date(
    tenant_id: str,
    iso_date: str,
    calendar_id: str = "",
) -> set[int]:
    """Consulta en tiempo real en Supabase y Google Calendar las horas ocupadas para una fecha específica."""
    booked_map = await _fetch_combined_booked_slots(tenant_id, calendar_id, iso_date, iso_date)
    return booked_map.get(iso_date, set())


def _format_day_slots_summary(iso_date: str, language: str, booked_hours: set[int]) -> str:
    """Genera el resumen conciso y aislado por fecha de horas ocupadas y libres en Supabase/Google Calendar."""
    all_hours = DEFAULT_MORNING_HOURS + DEFAULT_AFTERNOON_HOURS
    free_all = [h for h in all_hours if h not in booked_hours]

    if language == "es":
        if not booked_hours:
            return "TODO LIBRE de nueve de la mañana a cinco de la tarde"
        if not free_all:
            return f"SIN CUPOS DISPONIBLES EL {iso_date} (día lleno)"
        booked_words = [_format_hour_spoken(h, "es") for h in sorted(booked_hours)]
        return f"OCUPADO SOLO EL {iso_date}: {', '.join(booked_words)} | TODAS LAS DEMÁS HORAS DE 9AM A 5PM ESTÁN LIBRES"

    if not booked_hours:
        return "ALL SLOTS OPEN from nine AM to five PM"
    if not free_all:
        return f"FULLY BOOKED ON {iso_date} (no open slots)"
    booked_words_en = [_format_hour_spoken(h, "en") for h in sorted(booked_hours)]
    return f"BOOKED ONLY ON {iso_date}: {', '.join(booked_words_en)} | ALL OTHER HOURS 9AM TO 5PM ARE OPEN"


async def _insert_appointment_supabase(payload: dict) -> tuple[bool, str, dict | None]:
    """
    Inserta atómicamente la cita en public.spa_appointments de Supabase reutilizando el socket Keep-Alive (~110ms).
    Devuelve: (exito, codigo_estado_o_error, fila_creada)
    """
    sb_url, sb_key = _get_supabase_credentials()
    if not (sb_url and sb_key):
        logger.warning("⚠️ [SUPABASE INSERT] Credentials not configured in .env — simulated local booking only.")
        return True, "local_only", payload

    endpoint = f"{sb_url}/rest/v1/spa_appointments"
    headers = {
        "apikey": sb_key,
        "Authorization": f"Bearer {sb_key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }
    db_row = {
        "tenant_id": payload.get("tenant_id", DEFAULT_TENANT_ID),
        "client_name": payload["client_name"],
        "whatsapp_number": payload["whatsapp_number"],
        "email": payload["email"],
        "service_name": payload.get("service_name", DEFAULT_SERVICE_NAME),
        "appointment_date": payload["appointment_date"],
        "appointment_time": payload["appointment_time"],
        "appointment_at_iso": payload["appointment_at_iso"],
        "appointment_display": payload.get("appointment_display", ""),
        "language": payload.get("language", "es"),
        "call_direction": payload.get("call_direction", "outbound"),
        "status": "scheduled",
    }
    try:
        client = _get_supabase_http_client()
        resp = await client.post(endpoint, headers=headers, json=db_row)
        if resp.status_code in (200, 201):
            created_data = resp.json()
            row_obj = created_data[0] if isinstance(created_data, list) and created_data else db_row
            logger.info(
                "🗄️ [SUPABASE DB INSERT SUCCESS] id='%s' | date='%s' | time='%s' | reminder_24h='%s'",
                row_obj.get("id", "ok"),
                row_obj.get("appointment_date"),
                row_obj.get("appointment_time"),
                row_obj.get("reminder_trigger_at", "auto"),
            )
            return True, "created", row_obj
        if resp.status_code == 409 or "23505" in resp.text or "duplicate key" in resp.text.lower():
            logger.warning(
                "🚫 [SUPABASE SLOT COLLISION] Slot %s %s is already booked in database!",
                payload["appointment_date"],
                payload["appointment_time"],
            )
            return False, "slot_taken", None
        logger.error("❌ [SUPABASE INSERT ERROR] HTTP %d: %s", resp.status_code, resp.text[:300])
        return False, f"http_{resp.status_code}", None
    except Exception as exc:
        logger.error("❌ [SUPABASE INSERT EXCEPTION] %s", exc)
        return False, str(exc), None


async def _update_supabase_google_event_id(appointment_id: str, google_event_id: str) -> None:
    """Guarda el google_event_id en public.spa_appointments para habilitar reagendamientos y cancelaciones automáticas."""
    if not (appointment_id and google_event_id) or appointment_id == "N/A":
        return
    sb_url, sb_key = _get_supabase_credentials()
    if not (sb_url and sb_key):
        return

    endpoint = f"{sb_url}/rest/v1/spa_appointments"
    headers = {
        "apikey": sb_key,
        "Authorization": f"Bearer {sb_key}",
        "Content-Type": "application/json",
    }
    params = [("id", f"eq.{appointment_id}")]
    try:
        client = _get_supabase_http_client()
        resp = await client.patch(endpoint, headers=headers, params=params, json={"google_event_id": google_event_id})
        if resp.status_code in (200, 204):
            logger.info("🔗 [SUPABASE SYNC] Saved google_event_id='%s' for appointment='%s'", google_event_id, appointment_id)
        else:
            logger.warning("⚠️ [SUPABASE SYNC] HTTP %d updating google_event_id: %s", resp.status_code, resp.text[:200])
    except Exception as exc:
        logger.error("❌ [SUPABASE SYNC EXCEPTION] %s", exc)


async def _create_google_calendar_event(booking_payload: dict) -> str:
    """Pinta el bloque de 1 hora en el Google Calendar del Spa y vincula su ID en Supabase."""
    cal_id = (booking_payload.get("google_calendar_id") or os.environ.get("GOOGLE_CALENDAR_ID", DEFAULT_GOOGLE_CALENDAR_ID)).strip().strip("'\"")
    if not cal_id:
        logger.warning("⚠️ [GCAL SKIPPED] GOOGLE_CALENDAR_ID not configured — skipping calendar event creation.")
        return ""
    token = await _get_google_calendar_access_token()
    if not token:
        return ""

    start_iso = booking_payload.get("appointment_at_iso", "")
    try:
        start_dt = datetime.fromisoformat(start_iso)
        end_dt = start_dt + timedelta(hours=1)
        end_iso = end_dt.isoformat()
    except Exception:
        return ""

    business_name = booking_payload.get("business_name") or DEFAULT_BUSINESS_NAME
    service_name = booking_payload.get("service_name") or DEFAULT_SERVICE_NAME
    business_address = booking_payload.get("business_address") or DEFAULT_BUSINESS_ADDRESS
    client_name = booking_payload.get("client_name", "Cliente")
    whatsapp_num = booking_payload.get("whatsapp_number", "")
    client_email = booking_payload.get("email", "")
    appointment_id = booking_payload.get("appointment_id", "N/A")
    direction_label = "Llamada Saliente IA" if booking_payload.get("call_direction") == "outbound" else "Llamada Entrante IA"

    event_body = {
        "summary": f"{service_name} — {client_name}",
        "location": business_address,
        "description": (
            f"Cliente: {client_name}\n"
            f"Servicio: {service_name}\n"
            f"WhatsApp: {whatsapp_num}\n"
            f"Correo: {client_email}\n"
            f"Canal: {direction_label}\n"
            f"Negocio: {business_name}\n"
            f"Supabase ID: {appointment_id}"
        ),
        "start": {"dateTime": start_iso, "timeZone": "America/Caracas"},
        "end": {"dateTime": end_iso, "timeZone": "America/Caracas"},
        "colorId": "7",
    }

    endpoint = f"https://www.googleapis.com/calendar/v3/calendars/{quote(cal_id, safe='')}/events"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    try:
        client = _get_supabase_http_client()
        resp = await client.post(endpoint, headers=headers, json=event_body)
        if resp.status_code in (200, 201):
            ev_data = resp.json()
            ev_id = ev_data.get("id", "")
            logger.info(
                "📆 [GCAL EVENT CREATED] id='%s' | summary='%s' | start='%s'",
                ev_id,
                event_body["summary"],
                start_iso,
            )
            if ev_id and appointment_id != "N/A":
                await _update_supabase_google_event_id(appointment_id, ev_id)
            return ev_id
        logger.warning("⚠️ [GCAL CREATE ERROR] HTTP %d: %s", resp.status_code, resp.text[:250])
        return ""
    except Exception as exc:
        logger.error("❌ [GCAL CREATE EXCEPTION] %s", exc)
        return ""


def _build_google_calendar_client_link(booking_payload: dict) -> str:
    """Genera el enlace de 1 clic 'Agregar a mi Google Calendar' para el correo del cliente."""
    try:
        start_dt = datetime.fromisoformat(booking_payload["appointment_at_iso"]).astimezone(ZoneInfo("UTC"))
        end_dt = start_dt + timedelta(hours=1)
        dates_param = f"{start_dt.strftime('%Y%m%dT%H%M%SZ')}/{end_dt.strftime('%Y%m%dT%H%M%SZ')}"
        business_name = booking_payload.get("business_name") or DEFAULT_BUSINESS_NAME
        service_name = booking_payload.get("service_name") or DEFAULT_SERVICE_NAME
        address = booking_payload.get("business_address") or DEFAULT_BUSINESS_ADDRESS
        title = f"{service_name} — {business_name}"
        details = f"Cita confirmada en {business_name}. Servicio: {service_name}."
        return (
            "https://calendar.google.com/calendar/render?action=TEMPLATE"
            f"&text={quote(title)}"
            f"&dates={dates_param}"
            f"&details={quote(details)}"
            f"&location={quote(address)}"
        )
    except Exception:
        return ""


async def _send_resend_email(
    api_key: str,
    from_email: str,
    to_email: str,
    subject: str,
    html_body: str,
    reply_to: str = "",
    tag_label: str = "EMAIL",
) -> bool:
    """Envía un correo transaccional vía Resend REST API reutilizando el socket HTTP Keep-Alive."""
    payload: dict = {
        "from": from_email,
        "to": [to_email],
        "subject": subject,
        "html": html_body,
    }
    if reply_to:
        payload["reply_to"] = reply_to

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        client = _get_supabase_http_client()
        resp = await client.post("https://api.resend.com/emails", headers=headers, json=payload)
        if resp.status_code in (200, 201):
            data = resp.json()
            logger.info(
                "📧 [RESEND %s SUCCESS] id='%s' -> to='%s' | subject='%s'",
                tag_label,
                data.get("id", "ok"),
                to_email,
                subject,
            )
            return True
        logger.warning(
            "⚠️ [RESEND %s ERROR] HTTP %d sending to '%s': %s",
            tag_label,
            resp.status_code,
            to_email,
            resp.text[:250],
        )
        return False
    except Exception as exc:
        logger.error("❌ [RESEND %s EXCEPTION] to='%s': %s", tag_label, to_email, exc)
        return False


async def _dispatch_bilateral_email_notifications(booking_payload: dict) -> None:
    """Dispara en paralelo (0ms bloqueo de voz) el correo de confirmación al Cliente y la alerta operativa al Spa con fecha/hora ejecutiva."""
    resend_key = os.environ.get("RESEND_API_KEY", "").strip().strip("'\"")
    if not resend_key:
        logger.warning("⚠️ [RESEND SKIPPED] RESEND_API_KEY not found in .env — skipping email notifications.")
        return

    lang = booking_payload.get("language", "es")
    business_name = booking_payload.get("business_name") or DEFAULT_BUSINESS_NAME
    service_name = booking_payload.get("service_name") or DEFAULT_SERVICE_NAME
    business_address = booking_payload.get("business_address") or DEFAULT_BUSINESS_ADDRESS
    from_email = booking_payload.get("from_email") or os.environ.get("RESEND_FROM_EMAIL", DEFAULT_FROM_EMAIL).strip().strip("'\"")
    reply_to = booking_payload.get("reply_to_email") or os.environ.get("SPA_REPLY_TO_EMAIL", DEFAULT_REPLY_TO_EMAIL).strip().strip("'\"")
    admin_email = booking_payload.get("admin_email") or os.environ.get("SPA_ADMIN_EMAIL", DEFAULT_ADMIN_EMAIL).strip().strip("'\"")

    if "<" not in from_email and "@" in from_email:
        from_email = f"{business_name} <{from_email}>"

    client_name = booking_payload.get("client_name", "Cliente")
    client_email = booking_payload.get("email", "").strip()
    whatsapp_num = booking_payload.get("whatsapp_number", "")
    wa_clean_link = re.sub(r"\D", "", whatsapp_num)
    iso_date = booking_payload.get("appointment_date", "")
    iso_time = booking_payload.get("appointment_time", "")
    written_dt = booking_payload.get("appointment_written") or _format_written_datetime(iso_date, iso_time, lang)
    written_dt_es = _format_written_datetime(iso_date, iso_time, "es")
    appointment_id = booking_payload.get("appointment_id", "N/A")
    direction_label = "Llamada Saliente IA" if booking_payload.get("call_direction") == "outbound" else "Llamada Entrante IA"
    gcal_client_url = _build_google_calendar_client_link(booking_payload)
    gcal_btn_es = (
        f'<div style="text-align: center; margin: 22px 0;">'
        f'<a href="{gcal_client_url}" target="_blank" style="display: inline-block; background-color: #0d9488; color: #ffffff; text-decoration: none; font-weight: 600; font-size: 14px; padding: 11px 22px; border-radius: 8px;">📅 Agregar a mi Google Calendar</a>'
        f"</div>"
        if gcal_client_url
        else ""
    )
    gcal_btn_en = (
        f'<div style="text-align: center; margin: 22px 0;">'
        f'<a href="{gcal_client_url}" target="_blank" style="display: inline-block; background-color: #0d9488; color: #ffffff; text-decoration: none; font-weight: 600; font-size: 14px; padding: 11px 22px; border-radius: 8px;">📅 Add to my Google Calendar</a>'
        f"</div>"
        if gcal_client_url
        else ""
    )

    tasks = []

    # 1. Correo de Confirmación para el Cliente (ES / EN) con Fecha/Hora ejecutiva y botón de 1 clic para Google Calendar
    if client_email and "@" in client_email:
        if lang == "es":
            client_subject = f"✅ Cita Confirmada en {business_name} — {written_dt}"
            client_html = f"""
            <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 560px; margin: 0 auto; padding: 28px; border: 1px solid #e5e7eb; border-radius: 14px; background-color: #ffffff; color: #111827;">
                <h2 style="margin-top: 0; color: #0f766e;">¡Hola, {client_name}! Tu cita está confirmada</h2>
                <p style="font-size: 15px; line-height: 1.6; color: #374151;">
                    Te esperamos en <strong>{business_name}</strong>. Aquí tienes los detalles de tu reserva:
                </p>
                <div style="background-color: #f0fdfa; border-left: 4px solid #0d9488; padding: 16px; border-radius: 8px; margin: 20px 0;">
                    <p style="margin: 6px 0;"><strong>✨ Servicio:</strong> {service_name}</p>
                    <p style="margin: 6px 0;"><strong>📅 Fecha y Hora:</strong> {written_dt}</p>
                    <p style="margin: 6px 0;"><strong>📍 Dirección:</strong> {business_address}</p>
                    <p style="margin: 6px 0;"><strong>📱 WhatsApp registrado:</strong> {whatsapp_num}</p>
                </div>
                {gcal_btn_es}
                <p style="font-size: 14px; line-height: 1.5; color: #4b5563;">
                    🔔 <strong>Recordatorio automático:</strong> Te enviaremos un mensaje por WhatsApp 24 horas antes de tu cita para confirmar tu asistencia o ayudarte si necesitas reagendar.
                </p>
                <hr style="border: none; border-top: 1px solid #e5e7eb; margin: 24px 0;" />
                <p style="font-size: 12px; color: #9ca3af; margin-bottom: 0;">
                    Enviado automáticamente por el Asistente Inteligente de {business_name} · Powered by Plexus Code
                </p>
            </div>
            """
        else:
            client_subject = f"✅ Appointment Confirmed at {business_name} — {written_dt}"
            client_html = f"""
            <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 560px; margin: 0 auto; padding: 28px; border: 1px solid #e5e7eb; border-radius: 14px; background-color: #ffffff; color: #111827;">
                <h2 style="margin-top: 0; color: #0f766e;">Hi {client_name}, your appointment is confirmed!</h2>
                <p style="font-size: 15px; line-height: 1.6; color: #374151;">
                    We look forward to seeing you at <strong>{business_name}</strong>. Here are your booking details:
                </p>
                <div style="background-color: #f0fdfa; border-left: 4px solid #0d9488; padding: 16px; border-radius: 8px; margin: 20px 0;">
                    <p style="margin: 6px 0;"><strong>✨ Service:</strong> {service_name}</p>
                    <p style="margin: 6px 0;"><strong>📅 Date & Time:</strong> {written_dt}</p>
                    <p style="margin: 6px 0;"><strong>📍 Location:</strong> {business_address}</p>
                    <p style="margin: 6px 0;"><strong>📱 Registered WhatsApp:</strong> {whatsapp_num}</p>
                </div>
                {gcal_btn_en}
                <p style="font-size: 14px; line-height: 1.5; color: #4b5563;">
                    🔔 <strong>Automated Reminder:</strong> We'll send you a WhatsApp message 24 hours before your appointment so you can easily confirm or reschedule if needed.
                </p>
                <hr style="border: none; border-top: 1px solid #e5e7eb; margin: 24px 0;" />
                <p style="font-size: 12px; color: #9ca3af; margin-bottom: 0;">
                    Sent automatically by {business_name} AI Concierge · Powered by Plexus Code
                </p>
            </div>
            """
        tasks.append(
            _send_resend_email(
                api_key=resend_key,
                from_email=from_email,
                to_email=client_email,
                subject=client_subject,
                html_body=client_html,
                reply_to=reply_to or admin_email,
                tag_label="CLIENT_CONFIRMATION",
            )
        )

    # 2. Correo Operativo Interno para la Recepción / Dueño del Spa
    if admin_email and "@" in admin_email:
        admin_subject = f"🔔 Nueva Cita Agendada por IA: {client_name} ({written_dt_es})"
        admin_html = f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 580px; margin: 0 auto; padding: 26px; border: 1px solid #cbd5e1; border-radius: 14px; background-color: #f8fafc; color: #0f172a;">
            <h2 style="margin-top: 0; color: #1e293b;">🔔 Nueva Cita Registrada en {business_name}</h2>
            <p style="font-size: 14px; color: #475569;">
                El agente de voz IA ha verificado y registrado una nueva cita en Supabase y Google Calendar:
            </p>
            <div style="background-color: #ffffff; border: 1px solid #e2e8f0; padding: 16px; border-radius: 10px; margin: 16px 0;">
                <p style="margin: 6px 0;"><strong>👤 Cliente:</strong> {client_name}</p>
                <p style="margin: 6px 0;"><strong>📅 Fecha y Hora:</strong> {written_dt_es}</p>
                <p style="margin: 6px 0;"><strong>✨ Servicio:</strong> {service_name}</p>
                <p style="margin: 6px 0;"><strong>📱 WhatsApp:</strong> <a href="https://wa.me/{wa_clean_link}" style="color: #0284c7; text-decoration: none;">{whatsapp_num}</a></p>
                <p style="margin: 6px 0;"><strong>✉️ Correo:</strong> <a href="mailto:{client_email}" style="color: #0284c7; text-decoration: none;">{client_email}</a></p>
                <p style="margin: 6px 0;"><strong>📞 Canal:</strong> {direction_label} (Idioma: {lang.upper()})</p>
                <p style="margin: 6px 0;"><strong>🆔 Supabase ID:</strong> <code>{appointment_id}</code></p>
            </div>
            <p style="font-size: 12px; color: #64748b; margin-bottom: 0;">
                Notificación Operativa en Tiempo Real · Plexus Code AI Voice Engine
            </p>
        </div>
        """
        tasks.append(
            _send_resend_email(
                api_key=resend_key,
                from_email=from_email,
                to_email=admin_email,
                subject=admin_subject,
                html_body=admin_html,
                reply_to=client_email,
                tag_label="SPA_ADMIN_ALERT",
            )
        )

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def _dispatch_post_booking_integrations(booking_payload: dict) -> None:
    """Ejecuta en paralelo (0ms de impacto en voz) Resend Bilateral + Google Calendar del Spa + Sync en Supabase."""
    await asyncio.gather(
        _dispatch_bilateral_email_notifications(booking_payload),
        _create_google_calendar_event(booking_payload),
        return_exceptions=True,
    )


def _format_to_e164_ve(raw_input: str) -> str:
    """Convierte teléfonos de Venezuela (+58) o EE. UU./Canadá (+1) al estándar internacional E.164, limpiando tartamudeos y bloqueando '+0...'."""
    digits = re.sub(r"\D", "", raw_input)
    if not digits:
        return ""

    # Sanear posibles duplicaciones por autocorrección de voz en códigos de área venezolanos (ej. '042024...' -> '0424...')
    digits = re.sub(r"^(0?4)(?:20)(2[246]\d{7})$", r"\1\2", digits)
    digits = re.sub(r"^(0?4)(?:10)(1[246]\d{7})$", r"\1\2", digits)

    # 1. Números venezolanos (con o sin 0 inicial)
    if len(digits) == 11 and digits.startswith("0") and digits[1:].startswith(VE_AREA_PREFIXES):
        return f"+58{digits[1:]}"
    if len(digits) == 10 and digits.startswith(VE_AREA_PREFIXES):
        return f"+58{digits}"
    if len(digits) == 12 and digits.startswith("58") and digits[2:].startswith(VE_AREA_PREFIXES):
        return f"+{digits}"
    if len(digits) == 13 and digits.startswith("580") and digits[3:].startswith(VE_AREA_PREFIXES):
        return f"+58{digits[3:]}"
    # 2. Números de EE. UU. / Canadá (NANP: 10 dígitos [2-9]XX-XXX-XXXX o 11 dígitos iniciando en 1)
    if len(digits) == 10 and digits[0] in "23456789":
        return f"+1{digits}"
    if len(digits) == 11 and digits.startswith("1") and digits[1] in "23456789":
        return f"+{digits}"
    # 3. Otros números internacionales válidos (jamás permitir que empiecen con 0)
    if 11 <= len(digits) <= 15 and not digits.startswith("0"):
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
    """Extrae de forma determinista el teléfono en formato E.164 (+58 o +1), filtrando autocorrecciones como 'veinte... veinticuatro'."""
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
        # Filtro de autocorrección hablada: si dice 'veinte' e inmediatamente corrige a 'veinticuatro' (o 'twenty' -> 'twenty four'), ignorar el primero
        if tok in ("veinte", "twenty") and i + 1 < len(tokens):
            nxt_tok = tokens[i + 1]
            if nxt_tok.startswith("veint") or nxt_tok == "twenty":
                i += 1
                continue
        if tok in PHONE_TENS_TO_INT and i + 1 < len(tokens) and tokens[i + 1] == tok:
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
    Normaliza cualquier fecha y hora (ISO o en letras ES/EN, incluyendo mediodía/noon) para Supabase y TTS:
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

    # 4. Parsear la hora ("11:00", "12:00", "mediodía", "noon", "once de la mañana", "three in the afternoon")
    lower_t = raw_time.strip().lower()
    hour_24 = 9
    minute_val = 0

    if any(k in lower_t for k in ("mediodia", "mediodía", "noon", "midday")):
        hour_24 = 12
    else:
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
    spoken_time_str = _format_hour_spoken(hour_24, language)

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


def _get_temporal_context(
    language: str,
    custom_schedule: str = "",
    booked_slots_map: dict[str, set[int]] | None = None,
) -> str:
    """Pre-calcula en Python el calendario compacto de 14 días (horario corrido 9 AM a 5 PM) con aislamiento explícito por fecha."""
    now = datetime.now(ZoneInfo("America/Caracas"))
    period, greeting_phrase, farewell_phrase = _get_time_of_day_phrases(language)
    monday_this_week = now - timedelta(days=now.weekday())
    tomorrow = now + timedelta(days=1)
    day_after = now + timedelta(days=2)
    booked_map = booked_slots_map or {}

    this_week_items = []
    next_week_items = []
    free_tag = "TODO LIBRE 9am-5pm" if language == "es" else "ALL OPEN 9am-5pm"
    for d_idx in range(7):
        dt_this = monday_this_week + timedelta(days=d_idx)
        dt_next = monday_this_week + timedelta(weeks=1, days=d_idx)
        iso_this = dt_this.strftime("%Y-%m-%d")
        iso_next = dt_next.strftime("%Y-%m-%d")

        if dt_this.date() >= now.date():
            b_this = booked_map.get(iso_this, set())
            if b_this:
                this_week_items.append(
                    f"{_format_date_phonetic(dt_this, language)} ({iso_this}: {_format_day_slots_summary(iso_this, language, b_this)})"
                )
            else:
                this_week_items.append(f"{_format_date_phonetic(dt_this, language)} ({iso_this}: {free_tag})")

        b_next = booked_map.get(iso_next, set())
        if b_next:
            next_week_items.append(
                f"{_format_date_phonetic(dt_next, language)} ({iso_next}: {_format_day_slots_summary(iso_next, language, b_next)})"
            )
        else:
            next_week_items.append(f"{_format_date_phonetic(dt_next, language)} ({iso_next}: {free_tag})")

    hour_12 = now.hour % 12 or 12

    if language == "es":
        schedule_es = custom_schedule.strip() or (
            "Horario de atención corrido del negocio: de nueve de la mañana a cinco de la tarde (nueve, diez y once de la mañana; doce del mediodía; una, dos, tres, cuatro y cinco de la tarde). "
            "REGLA DE PRECISIÓN POR FECHA: Cada día tiene entre paréntesis su propio estado independiente. Si un día dice 'TODO LIBRE 9am-5pm', TODAS las horas de nueve de la mañana a cinco de la tarde están libres ese día (no confundas una hora ocupada de otro día con el día que pide el cliente)."
        )
        return (
            f"Ahora mismo (HOY) es {_format_date_phonetic(now, 'es')} [ISO: {now.strftime('%Y-%m-%d')}], {hour_12}:{now.minute:02d} {period}. "
            f"MAÑANA es {_format_date_phonetic(tomorrow, 'es')} [ISO: {tomorrow.strftime('%Y-%m-%d')}]. "
            f"PASADO MAÑANA es {_format_date_phonetic(day_after, 'es')} [ISO: {day_after.strftime('%Y-%m-%d')}]. "
            f"Días restantes de ESTA SEMANA: {' || '.join(this_week_items)}. "
            f"Días de LA PRÓXIMA SEMANA (semana entrante): {' || '.join(next_week_items)}. "
            f"{schedule_es} "
            f"Saludo actual: '{greeting_phrase}'. Despedida actual: 'que pases {farewell_phrase}'."
        )

    schedule_en = custom_schedule.strip() or (
        "Continuous business operating hours: nine in the morning to five in the afternoon (nine, ten, and eleven in the morning; twelve noon; one, two, three, four, and five in the afternoon). "
        "DATE ISOLATION RULE: Each day has its own independent availability in parentheses. If a day says 'ALL OPEN 9am-5pm', EVERY hour from nine AM to five PM is open on that day (never confuse a booked slot on another date with the date the prospect asked for)."
    )
    return (
        f"Right now (TODAY) is {_format_date_phonetic(now, 'en')} [ISO: {now.strftime('%Y-%m-%d')}], {hour_12}:{now.minute:02d} {period}. "
        f"TOMORROW is {_format_date_phonetic(tomorrow, 'en')} [ISO: {tomorrow.strftime('%Y-%m-%d')}]. "
        f"DAY AFTER TOMORROW is {_format_date_phonetic(day_after, 'en')} [ISO: {day_after.strftime('%Y-%m-%d')}]. "
        f"Remaining days THIS WEEK: {' || '.join(this_week_items)}. "
        f"Days NEXT WEEK: {' || '.join(next_week_items)}. "
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
    """Construye el motor TTS priorizando BYOK directo (Deepgram Aura-2 para Celeste ES y Thalia EN = 0 créditos Cartesia, o Cartesia BYOK)."""
    deepgram_api_key = os.environ.get("DEEPGRAM_API_KEY", "").strip()
    cartesia_api_key = os.environ.get("CARTESIA_API_KEY", "").strip()

    model = voice_id if voice_id else DEFAULT_TTS_MODEL
    voice: str | None = None
    if voice_id and ":" in voice_id:
        idx = voice_id.rfind(":")
        voice = voice_id[idx + 1:]
        model = voice_id[:idx]

    # 1. Si la voz seleccionada es de Deepgram (ej. Celeste 'deepgram/aura-2:celeste-es' o Thalia 'deepgram/aura-2:thalia-en'), usar Deepgram Direct Plugin (0 créditos de Cartesia)
    if model.startswith("deepgram/") and deepgram_api_key:
        lang_suffix = "es" if language == "es" else "en"
        default_v = "celeste" if language == "es" else "thalia"
        clean_voice = re.sub(r"-(es|en)$", "", (voice or default_v)).strip()
        dg_tts_model = f"aura-2-{clean_voice}-{lang_suffix}"
        logger.info(
            "🎙️ Deepgram Direct TTS (BYOK - 0 Cartesia credits) active: model='%s' | language='%s'",
            dg_tts_model,
            language,
        )
        return deepgram.TTS(model=dg_tts_model, api_key=deepgram_api_key)

    # 2. Si la voz seleccionada es de Cartesia (ej. voz clonada o 'cartesia/sonic-latest:...')
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
    """Precalienta recursos pesados (Silero VAD y compilador Pydantic de herramientas) en RAM antes de que entre la llamada."""
    proc.userdata["vad"] = silero.VAD.load(
        min_silence_duration=0.30,
        activation_threshold=0.65,
    )
    if lk_llm_utils is not None and hasattr(lk_llm_utils, "build_strict_openai_schema"):
        try:
            @llm.function_tool(description="warmup")
            async def _warmup_tool(a: str, b: int = 0) -> str:
                return a

            lk_llm_utils.build_strict_openai_schema(_warmup_tool)
        except Exception:
            pass


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
    pending_tasks: list[asyncio.Task] | None = None,
) -> None:
    """Monitorea en tiempo real el fin de la locución de despedida, asegura el envío de notificaciones en curso y corta la llamada."""
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

    if pending_tasks:
        active_tasks = [t for t in pending_tasks if not t.done()]
        if active_tasks:
            try:
                await asyncio.wait_for(asyncio.gather(*active_tasks, return_exceptions=True), timeout=5.0)
            except Exception:
                pass

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

    tenant_id: str = metadata.get("tenant_id") or os.environ.get("SPA_TENANT_ID", DEFAULT_TENANT_ID)
    business_name: str = metadata.get("business_name") or os.environ.get("SPA_BUSINESS_NAME", DEFAULT_BUSINESS_NAME)
    service_name: str = metadata.get("service_name") or os.environ.get("SPA_SERVICE_NAME", DEFAULT_SERVICE_NAME)
    business_address: str = metadata.get("business_address") or os.environ.get("SPA_ADDRESS", DEFAULT_BUSINESS_ADDRESS)
    from_email: str = metadata.get("from_email") or os.environ.get("RESEND_FROM_EMAIL", DEFAULT_FROM_EMAIL)
    reply_to_email: str = metadata.get("reply_to_email") or os.environ.get("SPA_REPLY_TO_EMAIL", DEFAULT_REPLY_TO_EMAIL)
    admin_email: str = metadata.get("admin_email") or os.environ.get("SPA_ADMIN_EMAIL", DEFAULT_ADMIN_EMAIL)
    google_calendar_id: str = (
        metadata.get("google_calendar_id")
        or os.environ.get("GOOGLE_CALENDAR_ID", DEFAULT_GOOGLE_CALENDAR_ID)
    ).strip().strip("'\"")

    phone_number: str = (
        metadata.get("phone_number")
        or os.environ.get("DEFAULT_CLIENT_PHONE", "+584241226796")
    )
    language: str = metadata.get("language") or os.environ.get("AGENT_LANGUAGE", "es")
    default_instructions = (
        (
            f"Representas a {business_name}. El objetivo es invitar al cliente a aprovechar su cincuenta por ciento de descuento "
            "en paquetes de depilación láser definitiva y agendarle una sesión de valoración diagnóstica gratuita de quince minutos "
            "con nuestra especialista para evaluar su piel sin compromiso."
        )
        if language == "es"
        else (
            f"You represent {business_name}. Your goal is to invite the client to use their pre-approved fifty percent discount "
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
    elif "f786b574-daa5-4673-aa0c-cbe3e8534c02" in voice_lower or "thalia" in voice_lower or "asteria" in voice_lower:
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
        f"Cliente frecuente y leal de {business_name}. Tiene pre-aprobado un cincuenta por ciento de descuento en depilación láser definitiva y valoración inicial sin costo."
        if language == "es"
        else f"Frequent and loyal client at {business_name}. Pre-approved for a fifty percent discount on laser hair removal and a free initial diagnostic assessment."
    )
    personalization_context: str = metadata.get("personalization_context") or default_context
    custom_business_hours: str = (
        metadata.get("business_hours")
        or os.environ.get("DEFAULT_BUSINESS_HOURS", "")
    )

    # Pre-Fetch asíncrono en paralelo: Supabase + Google Calendar del Spa para los próximos 14 días
    now_ve = datetime.now(ZoneInfo("America/Caracas"))
    monday_this_wk = now_ve - timedelta(days=now_ve.weekday())
    sunday_next_wk = monday_this_wk + timedelta(days=13)
    booked_slots_cache: dict[str, set[int]] = await _fetch_combined_booked_slots(
        tenant_id=tenant_id,
        calendar_id=google_calendar_id,
        start_iso_date=now_ve.strftime("%Y-%m-%d"),
        end_iso_date=sunday_next_wk.strftime("%Y-%m-%d"),
    )

    temporal_context = _get_temporal_context(language, custom_business_hours, booked_slots_cache)
    _, current_greeting, current_farewell = _get_time_of_day_phrases(language)
    raw_business_instructions = agent_instructions.strip()
    raw_call_context = personalization_context.strip()
    client_label = contact_name.strip() if contact_name else ("Cliente" if language == "es" else "Prospect")

    if language == "es":
        agent_instructions = (
            "<runtime_data>\n"
            f"  <temporal_awareness>{temporal_context}</temporal_awareness>\n"
            "  <business_and_client_context>\n"
            f"    <business_name>{business_name}</business_name>\n"
            f"    <business_address>{business_address}</business_address>\n"
            f"    <client_name>{client_label}</client_name>\n"
            f"    <client_phone>{phone_number or 'este mismo número'}</client_phone>\n"
            f"    <call_context>{raw_call_context}</call_context>\n"
            f"    <business_instructions>{raw_business_instructions}</business_instructions>\n"
            "  </business_and_client_context>\n"
            "</runtime_data>\n\n"
            "<agent_system_manifest>\n"
            "  <identity>\n"
            f"    Eres {agent_display_name}, especialista de atención al cliente en llamadas salientes de {business_name}. "
            "Asumes siempre que el cliente está ocupado haciendo otras cosas: tu ritmo es ágil, cercano, empático y ejecutivo. Hablas siempre en español neutro usando 'tú' (PROHIBIDO usar voseo argentino como 'usás', 'tenés' o 'podés'; di siempre 'usas', 'tienes', 'puedes') y JAMÁS uses términos en inglés como 'Senior Client Success Manager'. No recites plantillas de memoria; responde con naturalidad genuina.\n"
            "  </identity>\n\n"
            "  <execution_pipeline>\n"
            "    <phase index=\"1\" name=\"turno_1_presentacion_con_nombre_y_gancho_corto\">\n"
            f"      En cuanto el cliente confirme que es él después del 'Aló', salúdalo por su nombre y preséntate con tu nombre ({agent_display_name}) y la empresa ({business_name}), pon un PUNTO SEGUIDO, di el beneficio principal según <business_instructions> y cierra con un PUNTO SEGUIDO antes de la pregunta de permiso (máximo 25 palabras; ej. 'Hola, {client_label}, te habla {agent_display_name} de {business_name}. Te llamo porque tienes aprobado un cincuenta por ciento de descuento en depilación láser por ser cliente frecuente. ¿Tienes treinta segundos?').\n"
            "    </phase>\n"
            "    <phase index=\"2\" name=\"turno_2_el_valor_y_validacion_de_interes\">\n"
            "      Cuando el cliente diga 'Sí, dime', 'Sí, ok' o pregunte '¿De qué trata?', NO pidas fechas todavía. Abre con una muletilla natural unida con coma ('Pues mira, te incluimos...', 'Ah, mira, se trata de...'), explica brevemente la valoración sin compromiso según <business_instructions>, pon un PUNTO SEGUIDO y haz una pregunta corta para validar si le interesa aprovecharla.\n"
            "    </phase>\n"
            "    <phase index=\"3\" name=\"turno_3_compuerta_estricta_de_confirmacion_y_agenda_dinamica_supabase\">\n"
            "      - PASO 3.1 (COMPUERTA ESTRICTA DE CONFIRMACIÓN — PROHIBIDO ASIGNAR HOY O UNA HORA CUANDO EL CLIENTE SOLO DICE 'OK'):\n"
            "        * RAMA A (El cliente hace una pregunta, expresa duda, temor o dice 'pero'): Responde PRIMERO de forma directa, natural y empática a su pregunta exacta (por ejemplo: flexibilidad para escoger el día y hora que prefiera, gratuidad sin obligación de compra, o comodidad del láser frío sin dolor), sin pedirle días todavía y sin repetir metainstrucciones internas. Cierra con PUNTO SEGUIDO y pregunta: '¿Te animas a probarlo?'\n"
            "        * RAMA B (El cliente dice 'Ok', 'Bueno, está bien', 'Sí, dale' aceptando probar la valoración, pero AÚN NO HA DICHO QUÉ DÍA QUIERE): Está TERMINANTEMENTE PROHIBIDO inventar o asignarle el día de hoy o las nueve de la mañana por tu cuenta. Cuando el cliente solo dice 'Ok' para aceptar la valoración, pregúntale obligatoriamente: 'Perfecto. ¿Qué día de esta semana o de la próxima te queda bien?'\n"
            "      - PASO 3.2 (Dar Disponibilidad Real de Supabase en 2 Opciones SIN usar herramientas para esta semana o la próxima): Cuando el cliente mencione un día de esta semana o de la próxima, léelo directamente de <temporal_awareness> (NO invoques check_availability salvo que pida dos o más semanas en el futuro). Abre con una expresión humana natural ('A ver, déjame ver, el sábado...', 'Eeh, ya te digo, para el viernes...'), ofrece SOLO DOS opciones iniciales que figuren libres en <temporal_awareness> para esa fecha exacta, pon un PUNTO SEGUIDO y cierra con '. ¿Cuál de las dos te queda mejor?'\n"
            "      - PASO 3.3 (Selección de Hora, Mediodía/Otros Horarios y Verificación Exacta por Fecha en DB):\n"
            "        * SI EL CLIENTE ELIGE UNA HORA O PIDE CUALQUIER HORA ENTRE NUEVE DE LA MAÑANA Y CINCO DE LA TARDE (ej. 'A las cuatro', 'Al mediodía', 'A la una'): Revisa en <temporal_awareness> ÚNICAMENTE el paréntesis de ESA fecha exacta. Si esa fecha dice 'TODO LIBRE 9am-5pm' o si esa hora NO está listada en 'OCUPADO SOLO EL [fecha]' de ese mismo día, SIGNIFICA QUE SÍ ESTÁ LIBRE (aunque no la hayas nombrado en tu turno anterior). Acéptala de inmediato y responde con la estructura del PASO 4.1: 'Perfecto, entonces sería el [día y fecha sin año] [a la/a las según corresponda] [hora en letras]. Te envío la confirmación por WhatsApp y correo. ¿Usas este mismo número para tu WhatsApp?'\n"
            "        * SI PREGUNTA QUÉ OTROS HORARIOS TIENES EN LA MAÑANA, MEDIODÍA O TARDE: Abre con una expresión natural ('Eh, mira, en la tarde tengo...', 'Pues mira, también me queda...'), menciona DOS O TRES opciones libres de ese bloque según <temporal_awareness>, pon un PUNTO SEGUIDO y cierra con '¿Cuál te queda mejor?'\n"
            "        * ÚNICAMENTE SI LA HORA PEDIDA APARECE EXPLÍCITAMENTE EN 'OCUPADO SOLO EL [esa misma fecha]' O ES FUERA DE HORARIO (antes de las nueve o después de las cinco): Explícale el motivo y ofrécele dos horas libres cercanas cerrando con pregunta.\n"
            "    </phase>\n"
            "    <phase index=\"4\" name=\"confirmacion_obligatoria_paso_a_paso_whatsapp_y_correo\">\n"
            "      REGLA CRÍTICA INQUEBRANTABLE: SOLO entra a esta fase cuando el cliente YA haya elegido TANTO el DÍA como la HORA. Haz una sola pregunta corta por turno separando las oraciones con PUNTO SEGUIDO:\n"
            "      - PASO 4.1 (Confirmar Fecha/Hora Elegida sin Año y Preguntar WhatsApp): Apenas el cliente elija la hora (ej. 'A las cuatro', 'A la una'), di exactamente esta estructura (usando 'a la una' para la 1pm y 'a las' para las demás horas): 'Perfecto, entonces sería el [día y fecha sin año] [a la/a las] [hora en letras]. Te envío la confirmación por WhatsApp y correo. ¿Usas este mismo número para tu WhatsApp?'\n"
            "      - PASO 4.2 (Rama Sí / Rama No y Verificación en Bucle del WhatsApp con Pausas):\n"
            "        * Si confirma que usa ese mismo número: pídele su correo electrónico ('Listo, ¿a qué correo electrónico te la envío?').\n"
            "        * Si prefiere otro número: pregúntale a qué número de WhatsApp se la envías.\n"
            "        * VERIFICACIÓN OBLIGATORIA DEL TELÉFONO: Cuando te dicte el número, repítelo separando con comas los bloques de dígitos, pon un PUNTO SEGUIDO y haz la pregunta de confirmación ('Ah, perfecto, te repito el número: cero cuatro veinticuatro, uno dos dos, seis siete nueve seis. ¿Está bien así?'). PROHIBIDO decir 'mil', 'ciento' o 'cientos'. Si corrige algún dígito, ajústalo y vuelve a verificar hasta que confirme que SÍ. Recién ahí pídele el correo.\n"
            "      - PASO 4.3 (Verificación Fonética del Correo SIN símbolo @ ni dígitos): Cuando te dicte su correo, NO invoques book_appointment todavía. Repíteselo escribiendo SIEMPRE 'punto', 'arroba' y los números en letras, pon un PUNTO SEGUIDO y pregunta si está correcto ('Perfecto, confirmo tu correo: franlex punto ramírez dos arroba gmail punto com. ¿Es correcto?'). Espera a que confirme que SÍ.\n"
            "    </phase>\n"
            "    <phase index=\"5\" name=\"ejecucion_obligatoria_de_herramienta_book_appointment\">\n"
            f"      - ÚNICAMENTE después de completar el Paso 4.2 (WhatsApp) y de que el cliente responda 'Sí, correcto' a la lectura de su correo en el Paso 4.3, invoca OBLIGATORIAMENTE la herramienta book_appointment (pasando appointment_date en formato ISO YYYY-MM-DD y appointment_time en formato 24h HH:MM) y pronuncia el mensaje de despedida que te devuelva la herramienta.\n"
            f"      - Si en cualquier momento el cliente rechaza la llamada ('no me interesa', 'estoy ocupado'), despídete deseándole {current_farewell} e invoca end_call.\n"
            "    </phase>\n"
            "  </execution_pipeline>\n\n"
            "  <hard_constraints>\n"
            "    - PROHIBICIÓN ABSOLUTA DE INVENTAR FECHAS O SALTARSE PASOS: JAMÁS asignes una fecha u hora por tu cuenta cuando el cliente solo dice 'Ok' a la explicación del servicio (pregúntale siempre qué día le queda bien). Y JAMÁS digas 'ya quedó agendada' sin haber verificado antes WhatsApp y correo.\n"
            "    - EXPRESIONES HUMANAS REALES Y PROHIBICIÓN DE 'MMM': Cuando revises horarios o inicies una explicación, alterna con naturalidad frases como 'A ver, déjame ver,', 'A ver,', 'Eeh, ya te digo,', 'Eh, mira,', 'Bueno, déjame reviso, un segundito,' o 'Pues mira,' sin repetir la misma en turnos seguidos. PROHIBIDO escribir 'Mmm' o 'Hmm' (el sintetizador lo lee mal como 'Mim').\n"
            "    - PUNTUACIÓN PROSÓDICA EN 2 TIEMPOS (PAUSAS Y ENTONACIÓN DE PREGUNTA): Une la muletilla inicial con una COMA para arrancar fluido, pero SEPARA SIEMPRE cada oración y la pregunta final con un PUNTO SEGUIDO ('. ¿...?'). PROHIBIDO escribir una coma justo antes de '¿' (escribe SIEMPRE '. ¿Te gustaría?').\n"
            "    - PROHIBIDO MENCIONAR EL AÑO AL HABLAR: JAMÁS digas el año ('dos mil veintiséis' ni '2026') en tu respuesta de voz.\n"
            "    - ECONOMÍA VERBAL (CLIENTE OCUPADO): Ninguna respuesta debe superar las 28 palabras. Termina SIEMPRE cada turno (salvo la despedida final) con una pregunta clara precedida por un punto seguido.\n"
            "    - FONÉTICA ESTRICTA EN VOZ: Un solo párrafo continuo sin saltos de línea. PROHIBIDO usar signos de exclamación (¡ !) y PROHIBIDO escribir '@' en tus respuestas habladas. Escribe TODOS los números, fechas, porcentajes y horas en letras minúsculas. Al repetir teléfonos, separa los bloques con comas y JAMÁS pronuncies 'mil', 'ciento', 'cientos' ni 'millones'.\n"
            "    - AISLAMIENTO DE HERRAMIENTAS: JAMÁS escribas ni pronuncies nombres de funciones o código en tu respuesta.\n"
            "  </hard_constraints>\n"
            "</agent_system_manifest>"
        )
    else:
        agent_instructions = (
            "<runtime_data>\n"
            f"  <temporal_awareness>{temporal_context}</temporal_awareness>\n"
            "  <business_and_client_context>\n"
            f"    <business_name>{business_name}</business_name>\n"
            f"    <business_address>{business_address}</business_address>\n"
            f"    <client_name>{client_label}</client_name>\n"
            f"    <client_phone>{phone_number or 'this same number'}</client_phone>\n"
            f"    <call_context>{raw_call_context}</call_context>\n"
            f"    <business_instructions>{raw_business_instructions}</business_instructions>\n"
            "  </business_and_client_context>\n"
            "</runtime_data>\n\n"
            "<agent_system_manifest>\n"
            "  <identity>\n"
            f"    You are {agent_display_name}, a warm, personable client success specialist making an outbound call for {business_name}. "
            "Assume the prospect is busy: keep your tone natural, relaxed, friendly, and conversational. ALWAYS use natural spoken English contractions ('I'm', 'we're', 'there's', 'you'll', 'I'll', 'it's', 'don't', 'that's') — NEVER speak in stiff, uncontracted robotic sentences like 'I am calling' or 'There is no obligation'. Never recite memorized templates.\n"
            "  </identity>\n\n"
            "  <execution_pipeline>\n"
            "    <phase index=\"1\" name=\"turn_1_name_intro_and_short_hook\">\n"
            f"      As soon as the prospect confirms it is them after 'Hello?', greet them warmly by name and introduce yourself ({agent_display_name}) and the company ({business_name}), put a PERIOD, state their main benefit from <business_instructions> using natural contractions, put a PERIOD, and end with a quick permission question (max 25 words; e.g., 'Hi, {client_label}, this is {agent_display_name} from {business_name}. I'm calling because you've got a pre-approved fifty percent discount on laser hair removal as a frequent client. Do you have thirty seconds?').\n"
            "    </phase>\n"
            "    <phase index=\"2\" name=\"turn_2_value_and_interest_check\">\n"
            "      When the prospect says 'Sure / Go ahead' or asks 'What is this about?', DO NOT ask for dates yet. Open with a natural conversational transition ('So basically, we're including...', 'Well, we're offering...'), briefly explain the free assessment from <business_instructions>, put a PERIOD, and ask if they'd like to try it.\n"
            "    </phase>\n"
            "    <phase index=\"3\" name=\"turn_3_strict_confirmation_gate_and_dynamic_supabase_scheduling\">\n"
            "      - STEP 3.1 (STRICT CONFIRMATION GATE — NEVER AUTO-ASSIGN TODAY OR A TIME WHEN PROSPECT JUST SAYS 'OKAY'):\n"
            "        * BRANCH A (The prospect asks a question, expresses doubt, hesitation, or says 'but'): Answer their exact question FIRST in a warm, direct way using your own words (e.g., total flexibility to pick any day and time that works for them, 100% free with zero obligation, or painless cooling laser), without asking for days yet and without repeating internal prompt notes. Finish with a PERIOD and ask: 'Are you open to trying it?'\n"
            "        * BRANCH B (The prospect says 'Okay', 'Sure, let's do it' accepting the assessment, but HAS NOT NAMED A DAY YET): You are STRICTLY FORBIDDEN from inventing or assigning today's date or nine AM on your own. You MUST ask: 'Perfect. What day this week or next week works best for you?'\n"
            "      - STEP 3.2 (Offer 2 Real Open Options from <temporal_awareness> WITHOUT calling tools for this week or next week): When they mention a day this week or next week, read it directly from <temporal_awareness> (DO NOT call check_availability unless they ask for two or more weeks ahead). Open with a natural phrase ('Let's see, on Saturday...', 'Let me check real quick, for Friday...'), offer ONLY TWO initial options that are open in <temporal_awareness> for that exact date, put a PERIOD, and close with '. Which of those works better for you?'\n"
            "      - STEP 3.3 (Slot Selection, Noon/Other Open Slots & Exact Per-Date Check in DB):\n"
            "        * IF THE PROSPECT PICKS A TIME OR ASKS FOR ANY HOUR BETWEEN NINE AM AND FIVE PM (e.g., 'Four PM', 'Noon', 'One PM'): Check ONLY the parentheses for THAT exact date in <temporal_awareness>. If that date says 'ALL OPEN 9am-5pm' or if that hour is NOT listed under 'BOOKED ONLY ON [that date]', IT IS AVAILABLE (even if you didn't mention it in your previous turn). Accept it immediately and reply using STEP 4.1: 'Perfect, that'll be [day and date without year] at [time in words]. I'll send the confirmation via WhatsApp and email. Should we use this same number for WhatsApp?'\n"
            "        * IF THEY ASK WHAT OTHER MORNING, NOON, OR AFTERNOON TIMES ARE OPEN: Mention TWO OR THREE open slots for that block from <temporal_awareness>, put a PERIOD, and end with 'Which of those works best for you?'\n"
            "        * ONLY IF THE REQUESTED HOUR IS EXPLICITLY LISTED UNDER 'BOOKED ONLY ON [that exact date]' OR OUTSIDE HOURS (before nine AM or after five PM): Naturally explain why, put a PERIOD, and offer two close open times ending with a question.\n"
            "    </phase>\n"
            "    <phase index=\"4\" name=\"mandatory_step_by_step_whatsapp_and_email_verification\">\n"
            "      CRITICAL UNBREAKABLE RULE: ONLY enter this phase after the prospect has explicitly chosen BOTH a DAY and a TIME. Separate sentences with a PERIOD:\n"
            "      - STEP 4.1 (State Chosen Date/Time without Year & Ask WhatsApp): As soon as the time is chosen, use this exact structure: 'Perfect, that'll be [day and date without year] at [time in words]. I'll send the confirmation via WhatsApp and email. Should we use this same number for WhatsApp?'\n"
            "      - STEP 4.2 (WhatsApp Yes / No Branch & Natural Phone Read-Back Loop):\n"
            "        * If YES: ask what email address you should send it to ('Sounds good, what's the best email address to send that to?').\n"
            "        * If NO: ask what phone number is best for WhatsApp ('Sure thing, what's the best phone number for your WhatsApp?').\n"
            "        * MANDATORY PHONE VERIFICATION LOOP: When they dictate the number, read it back separating digit groups with commas, put a PERIOD, and ask for confirmation ('Got it, let me read that back to make sure: eight one three, seven six one, eight eight six seven. Does that sound right?'). NEVER say 'thousand' or 'hundred'. Once confirmed YES, ask for their email.\n"
            "      - STEP 4.3 (Phonetic Email Read-Back WITHOUT @ symbol or digits): When they give their email, DO NOT invoke book_appointment yet. Read it back spelling out 'dot', 'at', and numbers in words, put a PERIOD, and ask if it is correct ('Perfect, let me confirm your email: franlex dot ramirez two at gmail dot com. Is that right?'). Wait for their YES confirmation.\n"
            "    </phase>\n"
            "    <phase index=\"5\" name=\"mandatory_book_appointment_tool_execution\">\n"
            f"      - ONLY after completing Step 4.2 (WhatsApp) AND receiving the prospect's 'Yes' confirmation to the email read-back in Step 4.3, MANDATORILY invoke the book_appointment tool (passing appointment_date in ISO YYYY-MM-DD format and appointment_time in 24h HH:MM format) and speak the goodbye message returned by the tool.\n"
            f"      - If they reject the call, wish them {current_farewell} and invoke end_call.\n"
            "    </phase>\n"
            "  </execution_pipeline>\n\n"
            "  <hard_constraints>\n"
            "    - NEVER AUTO-ASSIGN DATES OR SKIP STEPS: NEVER invent or assign today's date or a time when the prospect just says 'Okay' to the assessment pitch (always ask what day works best for them first). And NEVER say 'you're confirmed' before verifying both WhatsApp and email.\n"
            "    - MANDATORY CONTRACTIONS & NATURAL ENGLISH FLOW: Always use contractions ('I'm', 'you're', 'we're', 'there's', 'it's', 'I'll', 'don't', 'what's'). Avoid robotic interjections like 'Uh, look,' or 'Ah, got it'; use natural American English transitions ('So basically,', 'Oh, totally,', 'Sure thing,', 'Gotcha,', 'Let's see,') without repeating the same opener in consecutive turns. NEVER write 'Mmm' or 'Hmm'.\n"
            "    - TWO-STAGE PROSODIC PUNCTUATION (NATURAL PAUSES & QUESTION INTONATION): Join your initial transition with a COMMA for a smooth start, but ALWAYS separate distinct sentences and the final question with a PERIOD ('. ...?'). NEVER put a comma right before the final question.\n"
            "    - NEVER MENTION THE YEAR OUT LOUD: NEVER say the year ('2026' or 'two thousand twenty-six') in your spoken reply.\n"
            "    - VERBAL ECONOMY (BUSY PROSPECT): Keep every reply under 28 words and ALWAYS end every turn (except the final goodbye) with a clear question preceded by a period.\n"
            "    - STRICT SPOKEN PHONETICS: Reply in a single continuous paragraph without line breaks. NEVER use exclamation marks (!) and NEVER write '@' in your spoken text. Spell out ALL numbers, phone digits, dates, percentages, and times in lowercase words. Separate phone digit groups with commas and NEVER read phone numbers as 'hundreds' or 'thousands'.\n"
            "    - TOOL ISOLATION: NEVER write or speak function names or code in your text response.\n"
            "  </hard_constraints>\n"
            "</agent_system_manifest>"
        )

    logger.info(
        "Starting call as '%s' (voice=%s, language=%s, tenant=%s, business='%s')",
        agent_display_name, voice_id or DEFAULT_TTS_MODEL, language, tenant_id, business_name,
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
            min_endpointing_delay=0.20,
            max_endpointing_delay=0.50,
            preemptive_generation=True,
            min_interruption_words=3,
            min_interruption_duration=0.6,
        )
    except TypeError:
        session = AgentSession(
            min_endpointing_delay=0.20,
            max_endpointing_delay=0.50,
            preemptive_generation=True,
        )

    call_ending = False
    pending_notification_tasks: list[asyncio.Task] = []
    booking_state: dict = {
        "last_user_text": "",
        "needs_readback_confirmation": False,
        "extracted_phone_digits": "",
        "force_remove_k_in_franlex": False,
        "last_checked_iso_date": "",
        "db_inserted": False,
    }

    @llm.function_tool(
        description=(
            "Queries real-time availability in Supabase and Google Calendar ONLY when the client asks for a date TWO OR MORE WEEKS in the future (weeks_ahead >= 2). "
            "DO NOT call this tool for today, tomorrow, this week (weeks_ahead=0), or next week (weeks_ahead=1) because their real-time availability is ALREADY loaded in <temporal_awareness>."
        )
    )
    async def check_availability(
        day_of_week: str,
        weeks_ahead: int = 2,
    ) -> str:
        now = datetime.now(ZoneInfo("America/Caracas"))
        clean_day = day_of_week.lower().strip()
        last_u_day = booking_state["last_user_text"].lower()

        if "pasado mañana" in clean_day or "day after tomorrow" in clean_day:
            target_dt = now + timedelta(days=2)
        elif "mañana" in clean_day or "tomorrow" in clean_day:
            target_dt = now + timedelta(days=1)
        elif "hoy" in clean_day or "today" in clean_day or "para hoy" in last_u_day or "es hoy" in last_u_day:
            target_dt = now
        else:
            matched_weekday: int | None = None
            for key, idx in WEEKDAY_MAP.items():
                if key in clean_day:
                    matched_weekday = idx
                    break
            if matched_weekday is None:
                matched_weekday = now.weekday()

            monday_current_week = now - timedelta(days=now.weekday())
            this_week_dt = monday_current_week + timedelta(days=matched_weekday)
            effective_weeks = max(0, int(weeks_ahead))
            if effective_weeks == 0 and this_week_dt.date() < now.date():
                effective_weeks = 1
            target_dt = monday_current_week + timedelta(weeks=effective_weeks, days=matched_weekday)

        req_iso = target_dt.strftime("%Y-%m-%d")
        booking_state["last_checked_iso_date"] = req_iso
        req_phonetic = _format_date_phonetic(target_dt, language)

        # Consultar disponibilidad real en caché o en Supabase + Google Calendar
        if req_iso in booked_slots_cache:
            booked_hours = booked_slots_cache[req_iso]
        else:
            booked_hours = await _fetch_booked_hours_for_date(tenant_id, req_iso, google_calendar_id)
            booked_slots_cache[req_iso] = booked_hours

        slots_summary = _format_day_slots_summary(req_iso, language, booked_hours)

        logger.info(
            "📅 [SUPABASE/GCAL CALENDAR TOOL] day='%s' weeks_ahead=%d -> %s (%s) | %s",
            day_of_week,
            weeks_ahead,
            req_phonetic,
            req_iso,
            slots_summary,
        )

        if language == "es":
            return (
                f"RESULTADO EXACTO DE AGENDA PARA {req_phonetic.upper()} (ISO: {req_iso}):\n"
                f"- Estado de cupos en base de datos y calendario: {slots_summary}.\n"
                f"- INSTRUCCIÓN DE RESPUESTA: Abre con una expresión humana natural ('A ver, déjame ver, el {req_phonetic}...', 'Eeh, ya te digo, el {req_phonetic}...'; PROHIBIDO escribir 'Mmm'), NO digas el año al hablar, ofrece primero SOLO DOS opciones que estén LIBRES (jamás ofrezcas horas ocupadas), pon un PUNTO SEGUIDO y cierra con una pregunta corta ('. ¿Cuál de las dos te queda mejor?')."
            )
        return (
            f"EXACT SCHEDULE RESULT FOR {req_phonetic.upper()} (ISO: {req_iso}):\n"
            f"- Real-time database & calendar slot status: {slots_summary}.\n"
            f"- RESPONSE INSTRUCTION: Open with a natural transition ('Let's see, on {req_phonetic}...' or 'Let me check real quick, on {req_phonetic}...'; NEVER write 'Mmm'), DO NOT say the year out loud, offer ONLY TWO initial options that are OPEN (never offer booked times), put a PERIOD, and end with a short question ('. Which of those works better for you?')."
        )

    @llm.function_tool(
        description=(
            "Formally books and persists the appointment in Supabase and Google Calendar ONLY AFTER reading back both the WhatsApp number (if a different one was given) and the email address "
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

        # Candado 1: Si aún no se ha pedido o proporcionado un correo electrónico válido con '@', bloquear e instruir Paso 4.1
        if not email or "no proporcionado" in email.lower() or "not provided" in email.lower() or "@" not in email:
            logger.info(
                "🛡️ [GUARDRAIL] Blocked premature book_appointment — missing valid email ('%s'). Forcing Step 4.1.",
                email,
            )
            if language == "es":
                return (
                    "ALTO - FALTAN DATOS OBLIGATORIOS (PASO 4): Aún no has verificado el WhatsApp ni pedido el correo electrónico del cliente. "
                    "PROHIBIDO decir que la cita ya quedó agendada. Aplica el PASO 4.1 ahora mismo: menciona el día y la hora elegidos y pregúntale: '¿Usas este mismo número para tu WhatsApp?'"
                )
            return (
                "STOP - MISSING REQUIRED DATA (STEP 4): You have not verified the prospect's WhatsApp or asked for their email address yet. "
                "Do NOT say the appointment is confirmed yet. Execute STEP 4.1 right now: state the chosen day and time and ask: 'Should we use this same number for WhatsApp?'"
            )

        # Candado 2 (ES/EN): si el cliente acaba de dictar o corregir teléfono/correo en este mismo turno, obligar a repetirlo y esperar confirmación
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
                    f"Según el protocolo obligatorio, primero debes repetirle el dato ('{email}') en palabras fonéticas, poner un punto seguido y preguntarle si está correcto. "
                    "Solo cuando el cliente te responda que sí en el siguiente turno, vuelve a invocar book_appointment."
                )
            return (
                f"STOP - DO NOT END THE CALL YET: The prospect just dictated or corrected contact info ('{last_u}'). "
                f"You must first read back the info ('{email}') in spoken words, put a period, and ask if it is correct. "
                "Only after they confirm yes on the next turn, invoke book_appointment again."
            )

        llm_e164 = _format_to_e164_ve(whatsapp_number)
        ext_e164 = booking_state["extracted_phone_digits"]
        if ext_e164 and (
            (ext_e164.startswith("+58") and len(ext_e164) == 13)
            or (ext_e164.startswith("+1") and len(ext_e164) == 12)
        ):
            resolved_phone = ext_e164
            logger.info(
                "📞 [PHONE PARSER] Deterministic Python E.164 phone applied: '%s' (LLM passed: '%s')",
                resolved_phone,
                whatsapp_number,
            )
        elif llm_e164:
            resolved_phone = llm_e164
            logger.info(
                "📞 [PHONE PARSER] Validated E.164 phone applied: '%s'",
                resolved_phone,
            )
        elif ext_e164:
            resolved_phone = ext_e164
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
        written_display = _format_written_datetime(iso_date, iso_time, language)
        time_prep_es = "a la" if spoken_time.startswith("una") else "a las"

        booking_payload = {
            "tenant_id": tenant_id,
            "business_name": business_name,
            "service_name": service_name,
            "business_address": business_address,
            "from_email": from_email,
            "reply_to_email": reply_to_email,
            "admin_email": admin_email,
            "google_calendar_id": google_calendar_id,
            "client_name": contact_name or ("Cliente" if language == "es" else "Prospect"),
            "appointment_date": iso_date,
            "appointment_time": iso_time,
            "appointment_at_iso": iso_full_at,
            "appointment_display": written_display,
            "appointment_written": written_display,
            "appointment_spoken": (
                f"{spoken_date} {time_prep_es} {spoken_time}"
                if language == "es"
                else f"{spoken_date} at {spoken_time}"
            ),
            "whatsapp_number": resolved_phone,
            "email": email.strip().lower(),
            "language": language,
            "call_direction": "outbound",
            "booked_at_iso": datetime.now(ZoneInfo("America/Caracas")).isoformat(),
        }

        # Inserción atómica en Supabase reutilizando el cliente HTTP Keep-Alive
        inserted_ok, status_reason, created_row = await _insert_appointment_supabase(booking_payload)
        if not inserted_ok and status_reason == "slot_taken":
            updated_booked = await _fetch_booked_hours_for_date(tenant_id, iso_date, google_calendar_id)
            booked_slots_cache[iso_date] = updated_booked
            new_summary = _format_day_slots_summary(iso_date, language, updated_booked)
            if language == "es":
                return (
                    f"ATENCIÓN - NO CUELGUES: El turno del {spoken_date} {time_prep_es} {spoken_time} acaba de ser ocupado en la base de datos. "
                    f"Horarios libres restantes para ese día: {new_summary}. "
                    "Disculpate con naturalidad diciendo que ese turno se acaba de ocupar y pregúntale cuál de los otros horarios disponibles prefiere."
                )
            return (
                f"WAIT - DO NOT HANG UP: The slot on {spoken_date} at {spoken_time} was just taken in the database. "
                f"Remaining open slots for that day: {new_summary}. "
                "Warmly let the client know that time slot just got booked and ask which of the remaining open times works best for them."
            )

        if isinstance(created_row, dict) and created_row.get("id"):
            booking_payload["appointment_id"] = created_row["id"]

        booking_state["db_inserted"] = True
        booked_hour_int = int(iso_time.split(":")[0])
        booked_slots_cache.setdefault(iso_date, set()).add(booked_hour_int)
        logger.info("✅ [BOOKING CONFIRMED & PERSISTED IN SUPABASE] %s", json.dumps(booking_payload, ensure_ascii=False))

        # Disparo asíncrono en paralelo: Resend Bilateral + Google Calendar del Spa + Sync google_event_id (0ms impacto en voz)
        notif_task = asyncio.create_task(_dispatch_post_booking_integrations(booking_payload))
        pending_notification_tasks.append(notif_task)

        if not call_ending:
            call_ending = True
            logger.info("📴 Booking completed — waiting for full confirmation & goodbye audio before hanging up…")
            asyncio.create_task(
                _terminate_call_session(
                    ctx,
                    session,
                    initial_wait=2.5,
                    max_wait=18.0,
                    pending_tasks=pending_notification_tasks,
                )
            )

        _, _, dynamic_farewell = _get_time_of_day_phrases(language)
        if language == "es":
            return (
                f"Cita guardada exitosamente en Supabase ({iso_date} {iso_time}). "
                f"Ahora despídete separando las ideas con PUNTO SEGUIDO y diciendo EXACTAMENTE así (SIN DECIR EL AÑO): "
                f"'Listo, {client_label}, tu cita para el {spoken_date} {time_prep_es} {spoken_time} ya quedó agendada. En segundos te llega la confirmación por WhatsApp y correo. Que pases {dynamic_farewell}.'"
            )
        return (
            f"Appointment saved in Supabase ({iso_date} {iso_time}). "
            f"Now say a warm, concise goodbye using contractions, separating sentences with a PERIOD, and stating EXACTLY: "
            f"'All set, {client_label}, you're confirmed for {spoken_date} at {spoken_time}. You'll get the confirmation via WhatsApp and email in just a few seconds. Have {dynamic_farewell}.'"
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
            lower_a = clean_text.lower()
            # Interceptor de seguridad: si por alucinación dice 'ya quedó agendada' sin haber insertado en Supabase, auto-corregir al Paso 4.1
            if not booking_state.get("db_inserted") and (
                "ya quedó agendada" in lower_a
                or "quedó agendada" in lower_a
                or "you're confirmed for" in lower_a
                or "appointment is confirmed" in lower_a
            ):
                logger.warning("🛡️ [AUTO-RECOVERY] Assistant spoke premature confirmation without DB insert — triggering Step 4.1 recovery.")
                recovery_instr = (
                    "Acabas de decir por error que la cita ya estaba agendada sin haber verificado antes el WhatsApp ni el correo del cliente. "
                    "Di inmediatamente: 'Disculpa, antes de cerrarla solo me falta confirmar tus datos para enviarte el comprobante. ¿Usas este mismo número para tu WhatsApp?'"
                    if language == "es"
                    else "You accidentally said the appointment was confirmed before verifying WhatsApp and email. Immediately say: 'Oh, sorry, before I lock that in, should we use this same number for your WhatsApp confirmation?'"
                )
                asyncio.create_task(session.generate_reply(instructions=recovery_instr))
            elif not call_ending and ("functions.end_call" in text or "end_call(" in text):
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