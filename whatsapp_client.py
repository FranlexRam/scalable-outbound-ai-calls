from abc import ABC, abstractmethod
import asyncio
import logging
import os
import re
from typing import Any

from dotenv import load_dotenv
import httpx

logger = logging.getLogger("outbound-agent")

DEFAULT_WHATSAPP_API_VERSION = "v25.0"
DEFAULT_TEMPLATE_CONFIRMATION_ES = "confirmacion_cita_v1"
DEFAULT_TEMPLATE_ADMIN_ES = "alerta_nueva_cita_v1"
DEFAULT_TEMPLATE_LANG_ES = "es"


def _normalize_wa_recipient(raw_phone: str) -> str:
    """Convierte un teléfono E.164 (ej. '+584241226796') al formato numérico puro que exige Meta ('584241226796')."""
    return re.sub(r"\D", "", (raw_phone or "").strip())


def _sanitize_template_param(value: Any, fallback: str = "No especificado") -> str:
    """Meta Cloud API rechaza saltos de línea, tabs, >4 espacios seguidos o strings vacíos dentro de parámetros de plantilla."""
    cleaned = re.sub(r"[\r\n\t]+", " ", str(value or "")).strip()
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned[:1000] if cleaned else fallback


class WhatsAppProvider(ABC):
    """Interfaz base (Patrón Adapter) para desacoplar el motor de voz del proveedor de WhatsApp (Meta Cloud API, Twilio, etc.)."""

    @abstractmethod
    async def send_template_message(
        self,
        to_phone: str,
        template_name: str,
        language_code: str,
        body_params: list[str],
        tag_label: str = "WHATSAPP",
    ) -> bool:
        """Envía una plantilla aprobada categoría Utility."""

    @abstractmethod
    async def send_text_message(
        self,
        to_phone: str,
        body_text: str,
        tag_label: str = "WHATSAPP_TEXT",
    ) -> bool:
        """Envía un mensaje de texto libre (válido dentro de la ventana de 24 horas)."""


class MetaCloudWhatsAppAdapter(WhatsAppProvider):
    """Cliente asíncrono directo contra Meta WhatsApp Cloud API reutilizando un httpx.AsyncClient Keep-Alive."""

    def __init__(
        self,
        phone_number_id: str,
        access_token: str,
        api_version: str = DEFAULT_WHATSAPP_API_VERSION,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.phone_number_id = phone_number_id.strip().strip("'\"")
        self.access_token = access_token.strip().strip("'\"")
        self.api_version = (api_version or DEFAULT_WHATSAPP_API_VERSION).strip().strip("'\"")
        self._http_client = http_client

    @property
    def is_configured(self) -> bool:
        return bool(self.phone_number_id and self.access_token)

    @property
    def _endpoint(self) -> str:
        return f"https://graph.facebook.com/{self.api_version}/{self.phone_number_id}/messages"

    async def _post_payload(self, payload: dict, tag_label: str, recipient: str) -> tuple[bool, int, str]:
        if not self.is_configured:
            logger.warning("⚠️ [WHATSAPP %s SKIPPED] Missing WHATSAPP_PHONE_NUMBER_ID or WHATSAPP_ACCESS_TOKEN.", tag_label)
            return False, 0, "missing_credentials"

        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }
        owns_client = self._http_client is None
        client = self._http_client or httpx.AsyncClient(timeout=8.0)
        try:
            resp = await client.post(self._endpoint, headers=headers, json=payload)
            if resp.status_code in (200, 201):
                data = resp.json()
                messages = data.get("messages", [])
                msg_id = messages[0].get("id", "ok") if messages else "ok"
                logger.info(
                    "💬 [WHATSAPP %s SUCCESS] id='%s' -> to='%s'",
                    tag_label,
                    msg_id,
                    recipient,
                )
                return True, resp.status_code, msg_id
            logger.warning(
                "⚠️ [WHATSAPP %s ERROR] HTTP %d sending to '%s': %s",
                tag_label,
                resp.status_code,
                recipient,
                resp.text[:300],
            )
            return False, resp.status_code, resp.text[:300]
        except Exception as exc:
            logger.error("❌ [WHATSAPP %s EXCEPTION] to='%s': %s", tag_label, recipient, exc)
            return False, 0, str(exc)
        finally:
            if owns_client:
                await client.aclose()

    async def send_template_message(
        self,
        to_phone: str,
        template_name: str,
        language_code: str,
        body_params: list[str],
        tag_label: str = "WHATSAPP",
    ) -> bool:
        recipient = _normalize_wa_recipient(to_phone)
        if not recipient:
            logger.warning("⚠️ [WHATSAPP %s SKIPPED] Invalid recipient phone: '%s'", tag_label, to_phone)
            return False

        payload: dict[str, Any] = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": recipient,
            "type": "template",
            "template": {
                "name": template_name.strip(),
                "language": {"code": language_code.strip()},
                "components": [
                    {
                        "type": "body",
                        "parameters": [
                            {"type": "text", "text": _sanitize_template_param(param)}
                            for param in body_params
                        ],
                    }
                ],
            },
        }
        ok, _, _ = await self._post_payload(payload, tag_label=tag_label, recipient=recipient)
        return ok

    async def send_text_message(
        self,
        to_phone: str,
        body_text: str,
        tag_label: str = "WHATSAPP_TEXT",
    ) -> bool:
        recipient = _normalize_wa_recipient(to_phone)
        if not recipient:
            return False

        payload: dict[str, Any] = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": recipient,
            "type": "text",
            "text": {
                "preview_url": False,
                "body": body_text.strip(),
            },
        }
        ok, _, _ = await self._post_payload(payload, tag_label=tag_label, recipient=recipient)
        return ok


async def dispatch_bilateral_whatsapp_notifications(
    booking_payload: dict,
    http_client: httpx.AsyncClient | None = None,
) -> None:
    """
    Dispara en paralelo por WhatsApp Cloud API (0ms de impacto en voz):
    1) Confirmación de cita al Cliente (plantilla Utility confirmacion_cita_v1).
    2) Alerta operativa al Dueño/Recepción (plantilla Utility alerta_nueva_cita_v1).
    """
    phone_number_id = (
        booking_payload.get("whatsapp_phone_number_id")
        or os.environ.get("WHATSAPP_PHONE_NUMBER_ID", "")
    ).strip().strip("'\"")
    access_token = (
        booking_payload.get("whatsapp_access_token")
        or os.environ.get("WHATSAPP_ACCESS_TOKEN", "")
    ).strip().strip("'\"")
    api_version = os.environ.get("WHATSAPP_API_VERSION", DEFAULT_WHATSAPP_API_VERSION).strip().strip("'\"")

    if not (phone_number_id and access_token):
        logger.warning("⚠️ [WHATSAPP SKIPPED] WHATSAPP_PHONE_NUMBER_ID or WHATSAPP_ACCESS_TOKEN not configured in .env.")
        return

    adapter = MetaCloudWhatsAppAdapter(
        phone_number_id=phone_number_id,
        access_token=access_token,
        api_version=api_version,
        http_client=http_client,
    )

    lang = (booking_payload.get("language") or "es").strip().lower()
    if lang == "en" and os.environ.get("WHATSAPP_TEMPLATE_CONFIRMATION_EN"):
        client_tpl = os.environ.get("WHATSAPP_TEMPLATE_CONFIRMATION_EN", "").strip().strip("'\"")
        client_tpl_lang = os.environ.get("WHATSAPP_TEMPLATE_LANG_EN", "en_US").strip().strip("'\"")
    else:
        client_tpl = (
            booking_payload.get("whatsapp_template_confirmation")
            or os.environ.get("WHATSAPP_TEMPLATE_CONFIRMATION_ES", DEFAULT_TEMPLATE_CONFIRMATION_ES)
        ).strip().strip("'\"")
        client_tpl_lang = os.environ.get("WHATSAPP_TEMPLATE_LANG_ES", DEFAULT_TEMPLATE_LANG_ES).strip().strip("'\"")

    admin_tpl = (
        booking_payload.get("whatsapp_template_admin")
        or os.environ.get("WHATSAPP_TEMPLATE_ADMIN_ES", DEFAULT_TEMPLATE_ADMIN_ES)
    ).strip().strip("'\"")
    admin_tpl_lang = os.environ.get("WHATSAPP_TEMPLATE_LANG_ES", DEFAULT_TEMPLATE_LANG_ES).strip().strip("'\"")

    client_name = booking_payload.get("client_name") or "Cliente"
    business_name = booking_payload.get("business_name") or "Zen Spa"
    service_name = booking_payload.get("service_name") or "Cita de Valoración"
    business_address = booking_payload.get("business_address") or "Dirección por confirmar"
    written_dt = booking_payload.get("appointment_written") or booking_payload.get("appointment_display") or ""
    client_phone = booking_payload.get("whatsapp_number") or ""
    client_email = booking_payload.get("email") or "No proporcionado"
    admin_phone = (
        booking_payload.get("whatsapp_admin_phone")
        or os.environ.get("WHATSAPP_ADMIN_PHONE", "")
    ).strip().strip("'\"")

    dev_freeform_fallback = os.environ.get("WHATSAPP_DEV_FREEFORM_FALLBACK", "0").strip() == "1"

    async def _send_client_confirmation() -> None:
        if not _normalize_wa_recipient(client_phone):
            return
        params = [client_name, business_name, service_name, written_dt, business_address]
        sent = await adapter.send_template_message(
            to_phone=client_phone,
            template_name=client_tpl,
            language_code=client_tpl_lang,
            body_params=params,
            tag_label="CLIENT_CONFIRMATION",
        )
        if not sent and dev_freeform_fallback:
            fallback_msg = (
                f"Confirmación de cita\n\n"
                f"Hola {client_name}, tu cita en {business_name} ha sido confirmada.\n\n"
                f"Servicio: {service_name}\n"
                f"Fecha y hora: {written_dt}\n"
                f"Dirección: {business_address}\n\n"
                f"Te enviaremos un recordatorio 24 horas antes. Si necesitas hacer algún cambio, responde a este mensaje."
            )
            await adapter.send_text_message(
                to_phone=client_phone,
                body_text=fallback_msg,
                tag_label="CLIENT_DEV_FALLBACK",
            )

    async def _send_admin_alert() -> None:
        if not _normalize_wa_recipient(admin_phone):
            return
        params = [business_name, client_name, service_name, written_dt, client_phone, client_email]
        sent = await adapter.send_template_message(
            to_phone=admin_phone,
            template_name=admin_tpl,
            language_code=admin_tpl_lang,
            body_params=params,
            tag_label="ADMIN_ALERT",
        )
        if not sent and dev_freeform_fallback:
            fallback_msg = (
                f"Nueva cita agendada por IA\n\n"
                f"Se ha registrado una nueva cita en {business_name}:\n\n"
                f"Cliente: {client_name}\n"
                f"Servicio: {service_name}\n"
                f"Fecha y hora: {written_dt}\n"
                f"WhatsApp: {client_phone}\n"
                f"Correo: {client_email}\n\n"
                f"La cita ya fue sincronizada en tu calendario y base de datos."
            )
            await adapter.send_text_message(
                to_phone=admin_phone,
                body_text=fallback_msg,
                tag_label="ADMIN_DEV_FALLBACK",
            )

    await asyncio.gather(
        _send_client_confirmation(),
        _send_admin_alert(),
        return_exceptions=True,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    load_dotenv(override=True)

    sample_booking = {
        "client_name": "Franlex",
        "business_name": os.environ.get("SPA_BUSINESS_NAME", "Zen Spa"),
        "service_name": os.environ.get("SPA_SERVICE_NAME", "Valoración Diagnóstica Láser (50% OFF)"),
        "business_address": os.environ.get(
            "SPA_ADDRESS",
            "Av. Principal, Centro Comercial Torre Zen, Piso 2, Local 24",
        ),
        "appointment_written": "Jueves, 26 de noviembre de 2026 · 01:00 PM",
        "whatsapp_number": os.environ.get("WHATSAPP_ADMIN_PHONE", "+584241226796"),
        "email": "franlex.ramirez2@gmail.com",
        "language": "es",
    }
    asyncio.run(dispatch_bilateral_whatsapp_notifications(sample_booking))