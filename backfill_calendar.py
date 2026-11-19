import os
from datetime import datetime, timedelta
from urllib.parse import quote
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2 import service_account
import httpx


def main() -> None:
    load_dotenv(override=True)

    sb_url = (os.environ.get("SUPABASE_URL") or os.environ.get("NEXT_PUBLIC_SUPABASE_URL") or "").strip().strip("'\"").rstrip("/")
    sb_key = (
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        or os.environ.get("SUPABASE_KEY")
        or os.environ.get("SUPABASE_ANON_KEY")
        or ""
    ).strip().strip("'\"")
    cal_id = os.environ.get("GOOGLE_CALENDAR_ID", "").strip().strip("'\"")
    sa_file = os.environ.get("GOOGLE_SERVICE_ACCOUNT_FILE", "google_service_account.json").strip().strip("'\"")
    business_name = os.environ.get("SPA_BUSINESS_NAME", "Zen Spa")
    business_address = os.environ.get("SPA_ADDRESS", "Av. Principal, Centro Comercial Torre Zen, Piso 2, Local 24")

    if not all([sb_url, sb_key, cal_id]):
        print("❌ Faltan variables en .env (SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY o GOOGLE_CALENDAR_ID).")
        return

    creds = service_account.Credentials.from_service_account_file(
        sa_file,
        scopes=["https://www.googleapis.com/auth/calendar"],
    )
    creds.refresh(GoogleAuthRequest())
    gcal_headers = {"Authorization": f"Bearer {creds.token}", "Content-Type": "application/json"}
    sb_headers = {"apikey": sb_key, "Authorization": f"Bearer {sb_key}", "Content-Type": "application/json"}

    tz = ZoneInfo("America/Caracas")
    with httpx.Client(timeout=10.0) as client:
        resp = client.get(
            f"{sb_url}/rest/v1/spa_appointments",
            headers=sb_headers,
            params=[
                ("google_event_id", "is.null"),
                ("status", "in.(scheduled,confirmed,rescheduled)"),
                ("select", "*"),
            ],
        )
        resp.raise_for_status()
        rows = resp.json()

        if not rows:
            print("✅ No hay citas pendientes por pintar (todas tienen google_event_id).")
            return

        print(f"🔎 Encontradas {len(rows)} citas previas sin google_event_id. Sincronizando...")
        cal_endpoint = f"https://www.googleapis.com/calendar/v3/calendars/{quote(cal_id, safe='')}/events"

        for row in rows:
            app_id = row["id"]
            raw_iso = row.get("appointment_at_iso")
            if raw_iso:
                start_dt = datetime.fromisoformat(str(raw_iso).replace("Z", "+00:00")).astimezone(tz)
            else:
                start_dt = datetime.strptime(
                    f"{row['appointment_date']} {str(row['appointment_time'])[:5]}",
                    "%Y-%m-%d %H:%M",
                ).replace(tzinfo=tz)
            end_dt = start_dt + timedelta(hours=1)

            service_name = row.get("service_name") or "Valoración Diagnóstica Láser (50% OFF)"
            client_name = row.get("client_name") or "Cliente"
            direction_label = "Llamada Saliente IA" if row.get("call_direction") == "outbound" else "Llamada Entrante IA"

            event_body = {
                "summary": f"{service_name} — {client_name}",
                "location": business_address,
                "description": (
                    f"Cliente: {client_name}\n"
                    f"Servicio: {service_name}\n"
                    f"WhatsApp: {row.get('whatsapp_number', '')}\n"
                    f"Correo: {row.get('email', '')}\n"
                    f"Canal: {direction_label}\n"
                    f"Negocio: {business_name}\n"
                    f"Supabase ID: {app_id}"
                ),
                "start": {"dateTime": start_dt.isoformat(), "timeZone": "America/Caracas"},
                "end": {"dateTime": end_dt.isoformat(), "timeZone": "America/Caracas"},
                "colorId": "7",
            }

            ev_resp = client.post(cal_endpoint, headers=gcal_headers, json=event_body)
            if ev_resp.status_code in (200, 201):
                ev_id = ev_resp.json().get("id", "")
                patch_resp = client.patch(
                    f"{sb_url}/rest/v1/spa_appointments",
                    headers=sb_headers,
                    params=[("id", f"eq.{app_id}")],
                    json={"google_event_id": ev_id},
                )
                print(
                    f"📆 Pintada: {row['appointment_date']} {str(row['appointment_time'])[:5]} "
                    f"-> gcal_id='{ev_id}' | Supabase patch HTTP {patch_resp.status_code}"
                )
            else:
                print(f"⚠️ Error pintando cita {app_id}: HTTP {ev_resp.status_code} - {ev_resp.text[:200]}")


if __name__ == "__main__":
    main()