import asyncio
import io
import json
import os
import re
import time
import uuid
import wave
from typing import Any, Dict, List, Optional

import aiohttp
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from livekit import api
from livekit.agents.inference import TTS as InferenceTTS
from pydantic import BaseModel, field_validator

load_dotenv(override=True)

app = FastAPI(title="Outbound Calling API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

E164_RE = re.compile(r"^\+[1-9]\d{7,14}$")

LIVEKIT_URL = os.environ["LIVEKIT_URL"]
LIVEKIT_API_KEY = os.environ["LIVEKIT_API_KEY"]
LIVEKIT_API_SECRET = os.environ["LIVEKIT_API_SECRET"]

# Nuevas variables SIP extraidas del .env
LIVEKIT_SIP_TRUNK_ID = os.environ["LIVEKIT_SIP_TRUNK_ID"]
TWILIO_PHONE_NUMBER = os.environ.get("TWILIO_PHONE_NUMBER", "+18133311309")

AGENT_NAME = "outbound-agent"
MAX_CAMPAIGN_CONTACTS = 500
CAMPAIGN_DISPATCH_DELAY = 1.5           # seconds between dispatches

# Candados de seguridad en LiveKit Cloud contra llamadas zombis (segundos)
ROOM_EMPTY_TIMEOUT = int(os.environ.get("ROOM_EMPTY_TIMEOUT", "30"))
ROOM_DEPARTURE_TIMEOUT = int(os.environ.get("ROOM_DEPARTURE_TIMEOUT", "15"))
MAX_CALL_DURATION_SECONDS = int(os.environ.get("MAX_CALL_DURATION_SECONDS", "180"))

# In-memory campaign store: campaign_id → campaign dict
# Each contact: {phone_number, personalization_context, status, room_name, error, dispatched_at}
campaigns: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lkapi() -> api.LiveKitAPI:
    return api.LiveKitAPI(
        url=LIVEKIT_URL,
        api_key=LIVEKIT_API_KEY,
        api_secret=LIVEKIT_API_SECRET,
    )


def _evict_old_campaigns() -> None:
    cutoff = time.time() - 86400  # 24 h
    stale = [cid for cid, c in campaigns.items() if c["created_at"] < cutoff]
    for cid in stale:
        campaigns.pop(cid, None)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class CallRequest(BaseModel):
    phone_number: str
    instructions: str = "You are a friendly assistant making a courtesy call. Be warm, concise, and helpful."
    agent_name: str = "AI Assistant"
    voice_id: Optional[str] = None
    language: Optional[str] = None  # "en" or "es"
    contact_name: str = ""
    personalization_context: str = ""

    @field_validator("phone_number")
    @classmethod
    def validate_e164(cls, v: str) -> str:
        if not E164_RE.match(v):
            raise ValueError("phone_number must be E.164 format (e.g. +12125551234)")
        return v


class WebCallRequest(BaseModel):
    instructions: str = "You are a friendly assistant making a courtesy call. Be warm, concise, and helpful."
    agent_name: str = "Daniela"
    voice_id: Optional[str] = None
    language: Optional[str] = "es"
    contact_name: str = "Franlex"
    personalization_context: str = ""


class CallResponse(BaseModel):
    success: bool
    room_name: str
    message: str


class CampaignContact(BaseModel):
    phone_number: str
    contact_name: str = ""
    personalization_context: str = ""

    @field_validator("phone_number")
    @classmethod
    def validate_e164(cls, v: str) -> str:
        if not E164_RE.match(v):
            raise ValueError(f"Invalid E.164 phone number: {v}")
        return v


class CampaignRequest(BaseModel):
    base_instructions: str
    agent_name: str = "AI Assistant"
    voice_id: Optional[str] = None
    language: Optional[str] = None  # "en" or "es"
    contacts: List[CampaignContact]


# ---------------------------------------------------------------------------
# Background task: dispatch campaign contacts one by one
# ---------------------------------------------------------------------------

async def _dispatch_campaign(
    campaign_id: str,
    base_instructions: str,
    agent_name: str,
    voice_id: Optional[str],
    language: Optional[str] = None,
) -> None:
    campaign = campaigns.get(campaign_id)
    if not campaign:
        return

    lk = _lkapi()
    try:
        for i, contact in enumerate(campaign["contacts"]):
            contact["status"] = "dispatching"
            room_name = f"camp-{campaign_id[:8]}-{i}-{uuid.uuid4().hex[:6]}"

            meta: dict = {
                "phone_number": contact["phone_number"],
                "agent_instructions": base_instructions,
                "agent_name": agent_name,
                "contact_name": contact.get("contact_name", ""),
                "personalization_context": contact["personalization_context"],
                "max_call_duration": MAX_CALL_DURATION_SECONDS,
            }
            if voice_id:
                meta["voice_id"] = voice_id
            if language:
                meta["language"] = language

            try:
                # 0. Pre-crear sala con autodestrucción automática en LiveKit Cloud
                await lk.room.create_room(
                    api.CreateRoomRequest(
                        name=room_name,
                        empty_timeout=ROOM_EMPTY_TIMEOUT,
                        departure_timeout=ROOM_DEPARTURE_TIMEOUT,
                    )
                )

                # 1. Despachar el Agente al mismo cuarto
                await lk.agent_dispatch.create_dispatch(
                    api.CreateAgentDispatchRequest(
                        agent_name=AGENT_NAME,
                        room=room_name,
                        metadata=json.dumps(meta),
                    )
                )

                # 2. Crear el participante SIP (Marcar el teléfono destino vía Twilio)
                await lk.sip.create_sip_participant(
                    api.CreateSIPParticipantRequest(
                        sip_trunk_id=LIVEKIT_SIP_TRUNK_ID,
                        sip_call_to=contact["phone_number"],
                        room_name=room_name,
                        participant_identity=f"sip_{contact['phone_number']}",
                        participant_name=f"Callee {contact['phone_number']}",
                    )
                )
                contact["status"] = "dispatched"
                contact["room_name"] = room_name
                contact["dispatched_at"] = time.time()
            except Exception as exc:
                contact["status"] = "error"
                contact["error"] = str(exc)

            if i < len(campaign["contacts"]) - 1:
                await asyncio.sleep(CAMPAIGN_DISPATCH_DELAY)
    finally:
        await lk.aclose()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/call", response_model=CallResponse)
async def initiate_call(req: CallRequest):
    room_name = f"call-{uuid.uuid4().hex[:12]}"

    meta: dict = {
        "phone_number": req.phone_number,
        "agent_instructions": req.instructions,
        "agent_name": req.agent_name,
        "contact_name": req.contact_name,
        "personalization_context": req.personalization_context,
        "max_call_duration": MAX_CALL_DURATION_SECONDS,
    }
    if req.voice_id:
        meta["voice_id"] = req.voice_id
    if req.language:
        meta["language"] = req.language

    lk = _lkapi()
    try:
        # 0. Pre-crear sala con autodestrucción automática en LiveKit Cloud
        await lk.room.create_room(
            api.CreateRoomRequest(
                name=room_name,
                empty_timeout=ROOM_EMPTY_TIMEOUT,
                departure_timeout=ROOM_DEPARTURE_TIMEOUT,
            )
        )

        # 1. Despachar el Agente al mismo cuarto
        await lk.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                agent_name=AGENT_NAME,
                room=room_name,
                metadata=json.dumps(meta),
            )
        )

        # 2. Crear el participante SIP (Marcar el teléfono destino vía Twilio)
        await lk.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                sip_trunk_id=LIVEKIT_SIP_TRUNK_ID,
                sip_call_to=req.phone_number,
                room_name=room_name,
                participant_identity=f"sip_{req.phone_number}",
                participant_name=f"Callee {req.phone_number}",
            )
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to dial or dispatch agent: {exc}") from exc
    finally:
        await lk.aclose()

    return CallResponse(
        success=True,
        room_name=room_name,
        message=f"Outbound call to {req.phone_number} dispatched. Room: {room_name}",
    )


@app.post("/web-call")
async def initiate_web_call(req: WebCallRequest):
    """Inicia una llamada de prueba 100% gratuita por WebRTC (Navegador) sin usar Twilio."""
    room_name = f"call-web-{uuid.uuid4().hex[:8]}"
    participant_identity = f"web_{uuid.uuid4().hex[:6]}"

    meta: dict = {
        "phone_number": "WEBRTC_BROWSER",
        "agent_instructions": req.instructions,
        "agent_name": req.agent_name,
        "contact_name": req.contact_name,
        "personalization_context": req.personalization_context,
        "max_call_duration": MAX_CALL_DURATION_SECONDS,
    }
    if req.voice_id:
        meta["voice_id"] = req.voice_id
    if req.language:
        meta["language"] = req.language

    lk = _lkapi()
    try:
        await lk.room.create_room(
            api.CreateRoomRequest(
                name=room_name,
                empty_timeout=ROOM_EMPTY_TIMEOUT,
                departure_timeout=ROOM_DEPARTURE_TIMEOUT,
            )
        )
        await lk.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                agent_name=AGENT_NAME,
                room=room_name,
                metadata=json.dumps(meta),
            )
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to start web call: {exc}") from exc
    finally:
        await lk.aclose()

    token = (
        api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity(participant_identity)
        .with_name(req.contact_name or "Web Tester")
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
            )
        )
        .to_jwt()
    )

    return {
        "success": True,
        "room_name": room_name,
        "livekit_url": LIVEKIT_URL,
        "token": token,
    }


@app.get("/active-calls")
async def list_active_calls():
    lk = _lkapi()
    try:
        resp = await lk.room.list_rooms(api.ListRoomsRequest())
        rooms = [
            {
                "room_name": r.name,
                "num_participants": r.num_participants,
                "created_at": r.creation_time,
            }
            for r in resp.rooms
            if r.name.startswith("call-") or r.name.startswith("camp-")
        ]
    finally:
        await lk.aclose()
    return {"rooms": rooms}


@app.delete("/active-calls/{room_name}")
async def stop_call(room_name: str):
    lk = _lkapi()
    try:
        await lk.room.delete_room(api.DeleteRoomRequest(room=room_name))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to stop call: {exc}") from exc
    finally:
        await lk.aclose()
    return {"stopped": room_name}


@app.post("/campaign")
async def launch_campaign(req: CampaignRequest, background_tasks: BackgroundTasks):
    if not req.contacts:
        raise HTTPException(status_code=400, detail="contacts list is empty")
    if len(req.contacts) > MAX_CAMPAIGN_CONTACTS:
        raise HTTPException(
            status_code=400,
            detail=f"Campaign limited to {MAX_CAMPAIGN_CONTACTS} contacts per batch",
        )

    _evict_old_campaigns()

    campaign_id = uuid.uuid4().hex
    campaigns[campaign_id] = {
        "created_at": time.time(),
        "base_instructions": req.base_instructions,
        "agent_name": req.agent_name,
        "voice_id": req.voice_id,
        "contacts": [
            {
                "phone_number": c.phone_number,
                "contact_name": c.contact_name,
                "personalization_context": c.personalization_context,
                "status": "pending",
                "room_name": None,
                "error": None,
                "dispatched_at": None,
            }
            for c in req.contacts
        ],
    }

    background_tasks.add_task(
        _dispatch_campaign,
        campaign_id,
        req.base_instructions,
        req.agent_name,
        req.voice_id,
        req.language,
    )

    return {"campaign_id": campaign_id, "total": len(req.contacts)}


@app.get("/campaign/{campaign_id}")
async def get_campaign(campaign_id: str):
    campaign = campaigns.get(campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")

    contacts = campaign["contacts"]
    counts: Dict[str, int] = {"pending": 0, "dispatching": 0, "dispatched": 0, "error": 0}
    for c in contacts:
        counts[c["status"]] = counts.get(c["status"], 0) + 1

    return {
        "campaign_id": campaign_id,
        "total": len(contacts),
        "counts": counts,
        "contacts": contacts,
    }


@app.get("/voice-preview")
async def voice_preview(
    voice_id: str = Query(..., description="Voice model string, e.g. cartesia/sonic-2:voice-uuid"),
    language: str = Query("en", description="Language code, e.g. en or es"),
):
    """Synthesize a short sample phrase with the given voice and return WAV audio."""
    if language == "es":
        sample_text = "¡Hola! Soy tu asistente de inteligencia artificial. ¿En qué puedo ayudarte hoy?"
    else:
        sample_text = "Hello! I'm your AI assistant. How can I help you today?"

    # Parse "provider/model:voice-id" → model + voice
    model = voice_id
    voice: Optional[str] = None
    if ":" in voice_id:
        idx = voice_id.rfind(":")
        voice = voice_id[idx + 1:]
        model = voice_id[:idx]

    if model.startswith("deepgram/") and voice:
        voice = re.sub(r"-(es|en)$", "", voice)
    elif model.startswith("rime/") and voice and voice.lower() == "serena":
        model = "rime/arcana"

    try:
        async with aiohttp.ClientSession() as http_session:
            tts_kwargs: Dict[str, Any] = {"model": model, "http_session": http_session}
            if voice:
                tts_kwargs["voice"] = voice
            if not model.startswith("rime/"):
                tts_kwargs["language"] = language

            tts = InferenceTTS(**tts_kwargs)

            audio_chunks: list[bytes] = []
            sample_rate = 24000

            try:
                stream = tts.synthesize(sample_text)
                async for event in stream:
                    audio_chunks.append(bytes(event.frame.data))
                    sample_rate = event.frame.sample_rate
            finally:
                await tts.aclose()

        pcm = b"".join(audio_chunks)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)   # 16-bit PCM
            wf.setframerate(sample_rate)
            wf.writeframes(pcm)
        buf.seek(0)
        return Response(content=buf.read(), media_type="audio/wav")

    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Voice preview failed: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)