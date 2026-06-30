import os
import json
import time
import uuid
import re
import base64
import subprocess
import threading
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory, session
from google import genai
from google.genai import types

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Image generation / editing model (Nano Banana 2 → fallback to Nano Banana).
# Override with IMAGE_GEN_MODELS env, e.g. "gemini-3-pro-image-preview,gemini-2.5-flash-image"
IMAGE_GEN_MODELS = os.environ.get(
    "IMAGE_GEN_MODELS",
    "gemini-3-pro-image-preview,gemini-2.5-flash-image,gemini-2.5-flash-image-preview",
).split(",")

TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
TWILIO_SERVICE_SID = os.environ.get("TWILIO_SERVICE_SID", "")

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
FRAMES_DIR = BASE_DIR / "static" / "frames"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
FRAMES_DIR.mkdir(parents=True, exist_ok=True)
PARTNER_DOCS_MAX_TOTAL = 10 * 1024 * 1024
PARTNER_DOC_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".pdf"}

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB
app.secret_key = os.environ.get("FLASK_SECRET", "society-inspector-dev-secret-change-me")

client = genai.Client(api_key=GEMINI_API_KEY)

# Reward points per issue severity.
POINTS_BY_SEVERITY = {"high": 15, "medium": 10, "low": 5}


def _points_for(severity: str | None) -> int:
    return POINTS_BY_SEVERITY.get((severity or "medium").lower(), 5)


# Reports persistence (phone -> list[report])
REPORTS_FILE = BASE_DIR / "data" / "reports.json"
REPORTS_FILE.parent.mkdir(parents=True, exist_ok=True)
_reports_lock = threading.Lock()
_reports_data: dict = {}


def _load_reports():
    global _reports_data
    if REPORTS_FILE.exists():
        try:
            with open(REPORTS_FILE) as f:
                payload = json.load(f) or {}
            if isinstance(payload, dict):
                _reports_data = payload
        except Exception:
            _reports_data = {}


def _save_reports():
    tmp = REPORTS_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(_reports_data, f)
    tmp.replace(REPORTS_FILE)


def _url_to_path(url: str | None) -> Path | None:
    if not url or not isinstance(url, str) or not url.startswith("/static/"):
        return None
    rel = url[len("/static/"):]
    return BASE_DIR / "static" / rel


def image_phash(path: Path | None) -> str | None:
    if not path or not path.exists():
        return None
    try:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
        dct = cv2.dct(np.float32(img))
        dct_low = dct[:8, :8]
        med = np.median(dct_low)
        bits = (dct_low > med).flatten()
        h = 0
        for b in bits:
            h = (h << 1) | int(b)
        return f"{h:016x}"
    except Exception:
        return None


def hamming_distance(a: str | None, b: str | None) -> int:
    if not a or not b or len(a) != len(b):
        return 999
    try:
        return bin(int(a, 16) ^ int(b, 16)).count("1")
    except ValueError:
        return 999


def text_jaccard(a: str | None, b: str | None) -> float:
    ta = set(re.findall(r"\w+", (a or "").lower()))
    tb = set(re.findall(r"\w+", (b or "").lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _build_timeline(submitted_ts: float) -> list:
    return [
        {"step": "submitted", "label": "Submitted", "ts": submitted_ts, "done": True, "current": False},
        {"step": "approval", "label": "Waiting for admin / moderator approval", "ts": None, "done": False, "current": True},
        {"step": "assigned", "label": "Service partner assigned", "ts": None, "done": False, "current": False},
        {"step": "started", "label": "Task started", "ts": None, "done": False, "current": False},
        {"step": "completed", "label": "Task complete", "ts": None, "done": False, "current": False},
    ]


def _earned_points_for(phone: str | None) -> int:
    if not phone:
        return 0
    with _reports_lock:
        return sum(int(r.get("points", 0)) for r in _reports_data.get(phone, []))


def _spent_points_for(phone: str | None) -> int:
    if not phone:
        return 0
    with _redemptions_lock:
        return sum(int(r.get("points_cost", 0)) for r in _redemptions_data.get(phone, []))


def _total_points_for(phone: str | None) -> int:
    return max(0, _earned_points_for(phone) - _spent_points_for(phone))


_load_reports()


# Reward redemption persistence (phone -> list of redemption rows)
REDEMPTIONS_FILE = BASE_DIR / "data" / "redemptions.json"
_redemptions_lock = threading.Lock()
_redemptions_data: dict = {}


def _load_redemptions():
    global _redemptions_data
    if REDEMPTIONS_FILE.exists():
        try:
            with open(REDEMPTIONS_FILE) as f:
                payload = json.load(f) or {}
            if isinstance(payload, dict):
                _redemptions_data = payload
        except Exception:
            _redemptions_data = {}


def _save_redemptions():
    tmp = REDEMPTIONS_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(_redemptions_data, f)
    tmp.replace(REDEMPTIONS_FILE)


_load_redemptions()


REWARDS_CATALOG = [
    {
        "id": "amazon-250",
        "name": "Amazon Voucher",
        "brand": "amazon",
        "amount_rs": 250,
        "points_cost": 50,
        "code": "TEST123",
    },
    {
        "id": "flipkart-100",
        "name": "Flipkart Voucher",
        "brand": "flipkart",
        "amount_rs": 100,
        "points_cost": 20,
        "code": "TEST123",
    },
]


# Reporter profiles persistence (phone -> reporter)
REPORTERS_FILE = BASE_DIR / "data" / "reporters.json"
_reporters_lock = threading.Lock()
_reporters_data: dict = {}


def _load_reporters():
    global _reporters_data
    if REPORTERS_FILE.exists():
        try:
            with open(REPORTERS_FILE) as f:
                payload = json.load(f) or {}
            if isinstance(payload, dict):
                _reporters_data = payload
        except Exception:
            _reporters_data = {}


def _save_reporters():
    tmp = REPORTERS_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(_reporters_data, f)
    tmp.replace(REPORTERS_FILE)


_load_reporters()


# Admin profiles persistence (phone -> admin)
ADMINS_FILE = BASE_DIR / "data" / "admins.json"
_admins_lock = threading.Lock()
_admins_data: dict = {}


def _load_admins():
    global _admins_data
    if ADMINS_FILE.exists():
        try:
            with open(ADMINS_FILE) as f:
                payload = json.load(f) or {}
            if isinstance(payload, dict):
                _admins_data = payload
        except Exception:
            _admins_data = {}


def _save_admins():
    tmp = ADMINS_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(_admins_data, f)
    tmp.replace(ADMINS_FILE)


_load_admins()


# Service partner profiles persistence (phone -> partner)
PARTNERS_FILE = BASE_DIR / "data" / "service_partners.json"
_partners_lock = threading.Lock()
_partners_data: dict = {}


def _load_partners():
    global _partners_data
    if PARTNERS_FILE.exists():
        try:
            with open(PARTNERS_FILE) as f:
                payload = json.load(f) or {}
            if isinstance(payload, dict):
                _partners_data = payload
        except Exception:
            _partners_data = {}


def _save_partners():
    tmp = PARTNERS_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(_partners_data, f)
    tmp.replace(PARTNERS_FILE)


_load_partners()

SECONDS_PER_YEAR = 365 * 24 * 60 * 60

PARTNER_TIME_SLOT_RANGES = {
    "Morning 8 AM - 12 PM": (8, 12),
    "Afternoon 12 PM - 4 PM": (12, 16),
    "Evening 4 PM - 8 PM": (16, 20),
    "Full day 8 AM - 8 PM": (8, 20),
}
WEEKDAY_ABBR = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _partner_slot_ranges(time_slots):
    out = []
    for s in time_slots or []:
        rng = PARTNER_TIME_SLOT_RANGES.get(s)
        if rng:
            out.append(rng)
    out.sort()
    # merge overlapping
    merged = []
    for r in out:
        if merged and r[0] <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], r[1]))
        else:
            merged.append(r)
    return merged


def _snap_to_partner_slot(start_epoch, weekdays, time_slots):
    """Return next epoch >= start_epoch that falls inside one of the partner's
    allowed weekday + time-slot windows. If we can't snap (no availability),
    return the original epoch."""
    valid_days = {d for d in (weekdays or []) if d in WEEKDAY_ABBR}
    ranges = _partner_slot_ranges(time_slots)
    if not valid_days or not ranges:
        return start_epoch
    dt = datetime.fromtimestamp(start_epoch).replace(second=0, microsecond=0)
    for _ in range(60):
        wd = WEEKDAY_ABBR[dt.weekday()]
        if wd in valid_days:
            for sh, eh in ranges:
                slot_start = dt.replace(hour=sh, minute=0, second=0, microsecond=0)
                slot_end = dt.replace(hour=eh, minute=0, second=0, microsecond=0)
                if dt <= slot_start:
                    return slot_start.timestamp()
                if dt < slot_end:
                    return dt.timestamp()
        dt = (dt + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return start_epoch


def _slot_label_for_hour(hour):
    for label, (sh, eh) in PARTNER_TIME_SLOT_RANGES.items():
        if sh <= hour < eh:
            return label
    return ""


def _next_available_label(epoch):
    if not epoch:
        return ""
    try:
        dt = datetime.fromtimestamp(float(epoch))
    except (TypeError, ValueError):
        return ""
    slot = _slot_label_for_hour(dt.hour)
    base = dt.strftime("%a, %d %b %Y · %I:%M %p").replace(" 0", " ")
    return f"{base}" + (f" ({slot})" if slot else "")


def _category_match(issue_category, partner_categories):
    """Return True if any word (3+ chars) overlaps between the issue category
    and any of the partner's categories."""
    if not issue_category or not partner_categories:
        return False
    a = {w for w in re.findall(r"[a-z]{3,}", issue_category.lower())}
    if not a:
        return False
    for pc in partner_categories:
        b = {w for w in re.findall(r"[a-z]{3,}", (pc or "").lower())}
        if a & b:
            return True
    return False


def _partner_next_available_at(partner, phone=None):
    val = partner.get("next_available_at")
    if not val:
        # Initialize lazily to "now" so legacy partners are usable.
        val = time.time()
    try:
        return float(val)
    except (TypeError, ValueError):
        return time.time()


def _estimate_duration_hours(issue):
    """Ask Gemini to estimate how many hours fixing this issue takes. Falls
    back to severity-based defaults if the model fails."""
    severity = (issue.get("severity") or "medium").lower()
    fallback = {"low": 2, "medium": 4, "high": 8}.get(severity, 4)
    try:
        prompt = (
            "Estimate the number of working hours a single service partner "
            "would need to fix this residential-society issue. Return ONLY a "
            'JSON object {"hours": <number>} with hours between 1 and 24.\n\n'
            f"Title: {issue.get('title') or ''}\n"
            f"Category: {issue.get('category') or ''}\n"
            f"Severity: {severity}\n"
            f"Detail: {issue.get('detail') or ''}\n"
        )
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt],
            config=types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json",
            ),
        )
        data = json.loads(response.text)
        hours = float(data.get("hours", fallback))
        if not (0.5 <= hours <= 48):
            hours = fallback
        return hours
    except Exception:
        return fallback


def _assignment_dict(phone, partner, scheduled_for, duration_hours, admin_name):
    return {
        "partner_phone": phone,
        "partner_name": partner.get("full_name", ""),
        "scheduled_for": scheduled_for,
        "duration_hours": duration_hours,
        "scheduled_for_label": _next_available_label(scheduled_for),
        "admin_name": admin_name,
        "ts": time.time(),
        "status": "pending",  # pending | accepted | denied | rescheduled (still active)
        "reschedule_count": 0,
        "history": [],  # list of {type: accept/deny/reschedule, sp_name, ts, ...}
    }


MAX_RESCHEDULES = 3
AUTO_APPROVE_LEAD_SECONDS = 2 * 3600


def _find_issue_in_report(report, issue_title):
    for it in report.get("issues") or []:
        title = it.get("title") or it.get("category") or "Issue"
        if title == issue_title:
            return it
    return None


def _refresh_assigned_timeline(report):
    """Ensure timeline 'assigned' step reflects current assignments."""
    timeline = report.get("timeline") or []
    if not timeline:
        return
    assignments = report.get("assignments") or {}
    any_assigned = bool(assignments)
    earliest_ts = None
    for a in assignments.values():
        try:
            ts = float(a.get("scheduled_for") or 0)
        except (TypeError, ValueError):
            ts = 0
        if ts and (earliest_ts is None or ts < earliest_ts):
            earliest_ts = ts
    decided = bool(report.get("approved_issues") or report.get("denied_issues"))
    for step in timeline:
        if step.get("step") == "assigned":
            step["done"] = any_assigned
            step["current"] = decided and not any_assigned
            step["ts"] = earliest_ts if any_assigned else None
        elif step.get("step") == "approval":
            if decided:
                step["done"] = True
                step["current"] = False


def _partner_doc_url(phone, stored_filename):
    if not phone or not stored_filename:
        return ""
    return f"/partner/docs/{urllib.parse.quote(stored_filename)}"


def _with_partner_experience(partner, phone=None):
    if not partner:
        return partner
    out = dict(partner)
    try:
        baseline = int(out.get("experience_years_at_signup", out.get("experience_years", 0)) or 0)
    except (TypeError, ValueError):
        baseline = 0
    try:
        created_at = float(out.get("created_at") or time.time())
    except (TypeError, ValueError):
        created_at = time.time()
    elapsed_years = max(0, int((time.time() - created_at) // SECONDS_PER_YEAR))
    out["experience_years_at_signup"] = baseline
    out["experience_years"] = baseline + elapsed_years
    docs = []
    for doc in out.get("verification_documents") or []:
        item = dict(doc)
        if phone and item.get("stored_filename") and not item.get("url"):
            item["url"] = _partner_doc_url(phone, item.get("stored_filename"))
        docs.append(item)
    out["verification_documents"] = docs
    nxt = out.get("next_available_at")
    if not nxt:
        nxt = out.get("created_at") or time.time()
        out["next_available_at"] = nxt
    out["next_available_label"] = _next_available_label(nxt)
    return out


def _filestorage_size(file_obj):
    try:
        pos = file_obj.stream.tell()
        file_obj.stream.seek(0, os.SEEK_END)
        size = file_obj.stream.tell()
        file_obj.stream.seek(pos)
        return size
    except Exception:
        return int(file_obj.content_length or 0)


def _validate_partner_docs(files):
    total_size = 0
    for doc in files:
        ext = Path(doc.filename or "").suffix.lower()
        mime = (doc.mimetype or "").lower()
        is_allowed = mime.startswith("image/") or mime == "application/pdf" or ext in PARTNER_DOC_EXTS
        if not is_allowed:
            return "Upload images or PDFs only"
        total_size += _filestorage_size(doc)
    if total_size > PARTNER_DOCS_MAX_TOTAL:
        return "Documents must be 10 MB total or less"
    return ""


def _save_partner_docs(phone, files):
    saved_docs = []
    if not files:
        return saved_docs
    safe_phone = re.sub(r"[^A-Za-z0-9_-]", "_", phone or uuid.uuid4().hex)
    docs_dir = UPLOAD_DIR / "partner_docs" / safe_phone
    docs_dir.mkdir(parents=True, exist_ok=True)
    for doc in files:
        original = doc.filename or "document"
        safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", original)[:120] or "document"
        stored_name = f"{uuid.uuid4().hex[:10]}_{safe_name}"
        size = _filestorage_size(doc)
        doc.save(docs_dir / stored_name)
        saved_docs.append({
            "filename": original,
            "stored_filename": stored_name,
            "size": size,
            "mime": doc.mimetype or "",
            "url": _partner_doc_url(phone, stored_name),
        })
    return saved_docs


def _url_to_path(url):
    if not url or not isinstance(url, str) or not url.startswith("/static/"):
        return None
    rel = url[len("/static/"):]
    return BASE_DIR / "static" / rel


def _area_words(text):
    return {w for w in re.findall(r"[a-z0-9]{3,}", (text or "").lower())}


def _location_in_admin_area(loc, admin_area):
    a = _area_words(admin_area)
    l = _area_words(loc)
    if not a or not l:
        return False
    return bool(a & l)


def _with_issue_statuses(report):
    out = dict(report)
    denied_by_title = {
        (d.get("issue_title") or ""): d
        for d in (report.get("denied_issues") or [])
        if d.get("issue_title")
    }
    issues = []
    for issue in report.get("issues") or []:
        item = dict(issue)
        title = item.get("title") or item.get("category") or "Issue"
        denial = denied_by_title.get(title)
        if denial:
            item["issue_status"] = "denied"
            item["status_label"] = "Denied by admin"
            item["denial"] = dict(denial)
        else:
            item["issue_status"] = "waiting_for_approval"
            item["status_label"] = "Wait for admin approval"
        issues.append(item)
    out["issues"] = issues
    if denied_by_title:
        out["status_label"] = f"{len(denied_by_title)} issue{'s' if len(denied_by_title) != 1 else ''} denied by admin"
    return out


def _attach_reporter_info(report):
    out = _with_issue_statuses(report)
    ph = report.get("phone")
    rep = _reporters_data.get(ph) if ph else None
    if rep:
        out["reporter_name"] = rep.get("full_name", "")
        out["reporter_address"] = rep.get("address", "")
        out["reporter_email"] = rep.get("email", "")
    return out


def _find_report(report_id):
    with _reports_lock:
        for ph, reports in _reports_data.items():
            for r in reports:
                if r.get("id") == report_id:
                    return ph, r
    return None, None


ADMIN_CATEGORIES = [
    "Infrastructure", "Waste Disposal", "Water & Drainage",
    "Street Lighting", "Roads & Potholes", "Public Safety", "Parks & Greenery",
]


# In-memory store for background analysis jobs.
# job_id -> {status: running|done|failed, phone, started_at, result, error}
_analysis_jobs: dict = {}
_analysis_lock = threading.Lock()


def _get_running_job_for_phone(phone: str | None) -> str | None:
    if not phone:
        return None
    with _analysis_lock:
        for jid, j in _analysis_jobs.items():
            if j.get("phone") == phone and j.get("status") == "running":
                return jid
    return None

ISSUE_CATEGORIES = """
- Maintenance & Infrastructure: poor upkeep of lobbies/staircases/lifts, irregular cleaning, lift breakdowns, leaking roofs/walls/pipelines, damaged flooring, peeling paint, non-functional lighting, broken gym equipment, neglected pool, poor play area upkeep
- Water & Power: irregular water supply, contamination/hardness, power cuts, generator fuel issues, billing disputes
- Parking: insufficient slots, unauthorized parking, double parking, no two-wheeler spots, EV charger disputes
- Security: missing/broken CCTV, untrained guards, unauthorized access, no visitor management, theft, no police verification
- Waste Management: improper segregation, overflowing dustbins, illegal dumping, no composting/recycling
- Financial & Administrative: opaque fund use, embezzlement, defaulters, no audit, arbitrary fee hikes, sinking fund mismanagement, delayed deposit refunds
- Governance & Committee: biased decisions, no AGMs, no accountability, bylaw non-compliance, favoritism, election confusion
- Communication: no official channel, last-minute notices, language barriers, missed announcements, no grievance tracking
- Resident Behavior & Disputes: noise complaints, pet disputes, littering, smoking, amenity conflicts, tenant/owner tensions, neighbor disputes, illegal modifications
- Housekeeping & Staff: staff absenteeism, untrained/underpaid staff, no vendor accountability, salary disputes
- Legal & Compliance: fire safety violations, illegal commercial use, unregistered society, ownership disputes, short-term rental issues
- Environment & Amenities: no green spaces, dying gardens, air/noise pollution, pest infestations, stray animals, unused rainwater/solar systems
- Social & Community: low participation, elderly isolation, discrimination, no events, owner/tenant rifts
"""

ANALYSIS_PROMPT = f"""You are analyzing a video walkthrough of a residential housing society / apartment complex.

Your tasks:
1. Identify distinct LOCATIONS visible in the video (e.g., lift lobby, staircase, main door, parking, corridor, garden, gym, rooftop, basement, entrance gate, garbage area, room, kitchen, balcony, water tank area, electrical room, security cabin).
2. For each location, pick the SINGLE best representative timestamp (in seconds, decimal allowed) where it is most clearly visible.
3. Detect any ISSUES visible at that location, mapped to the categories below. Only flag what is actually visible.

Issue categories to consider:
{ISSUE_CATEGORIES}

Return ONLY valid JSON (no markdown fences, no commentary) in this exact schema:
{{
  "locations": [
    {{
      "name": "short location name",
      "timestamp": 12.5,
      "description": "what is visible at this location",
      "issues": [
        {{
          "category": "one of the categories above (top-level name)",
          "title": "short issue title",
          "detail": "what specifically is wrong and where in the frame",
          "severity": "low | medium | high"
        }}
      ]
    }}
  ],
  "summary": "1-2 sentence overall assessment of the property condition"
}}

If no issues are visible at a location, return an empty issues array for it.
"""


def extract_frame(video_path: Path, timestamp_sec: float, out_path: Path) -> bool:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = total_frames / fps if fps else 0
    ts = max(0.0, min(float(timestamp_sec), max(duration - 0.05, 0.0)))

    # try msec seek first, then frame seek
    cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(int(ts * fps), max(int(total_frames) - 1, 0)))
        ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return False
    cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return True


def video_duration(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return float(frames / fps) if fps else 0.0


def annotate_issue_image(src_path: Path, title: str, detail: str, out_path: Path) -> bool:
    try:
        with open(src_path, "rb") as f:
            data = f.read()
        image_part = types.Part.from_bytes(data=data, mime_type="image/jpeg")
        prompt = (
            "Edit this photo: draw ONE thick bright GREEN rectangular border (outline only, "
            "no fill, ~6 pixels thick) tightly around the area that shows the following issue. "
            "Keep the rest of the image perfectly identical — same colors, lighting, content, "
            "framing. Return the modified image only.\n\n"
            f"Issue title: {title}\n"
            f"Issue detail: {detail}"
        )
        for model_id in IMAGE_GEN_MODELS:
            model_id = model_id.strip()
            if not model_id:
                continue
            try:
                response = client.models.generate_content(
                    model=model_id,
                    contents=[image_part, prompt],
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE", "TEXT"],
                    ),
                )
            except Exception:
                continue
            for cand in (response.candidates or []):
                content = getattr(cand, "content", None)
                for part in (getattr(content, "parts", None) or []):
                    inline = getattr(part, "inline_data", None)
                    if inline and getattr(inline, "data", None):
                        with open(out_path, "wb") as f:
                            f.write(inline.data)
                        return True
        return False
    except Exception:
        return False


def trim_video_clip(src: Path, start: float, end: float, out_path: Path) -> bool:
    if end <= start:
        return False
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-ss", f"{max(0.0, start):.2f}",
                "-to", f"{end:.2f}",
                "-i", str(src),
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
                "-c:a", "aac", "-b:a", "96k",
                "-movflags", "+faststart",
                "-pix_fmt", "yuv420p",
                str(out_path),
            ],
            capture_output=True,
            timeout=90,
        )
        return result.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0
    except Exception:
        return False


def parse_json_response(text: str) -> dict:
    text = text.strip()
    # strip code fences if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    # find the first { ... } block defensively
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start : end + 1]
    return json.loads(text)


def analyze_video_with_gemini(video_path: Path) -> dict:
    uploaded = client.files.upload(file=str(video_path))
    # wait until processed
    while uploaded.state and uploaded.state.name == "PROCESSING":
        time.sleep(2)
        uploaded = client.files.get(name=uploaded.name)
    if uploaded.state and uploaded.state.name == "FAILED":
        raise RuntimeError("Gemini failed to process the video upload.")

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[uploaded, ANALYSIS_PROMPT],
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
        ),
    )

    try:
        client.files.delete(name=uploaded.name)
    except Exception:
        pass

    return parse_json_response(response.text)


def twilio_post(path: str, data: dict) -> tuple[int, dict]:
    url = f"https://verify.twilio.com/v2/Services/{TWILIO_SERVICE_SID}/{path}"
    body = urllib.parse.urlencode(data).encode()
    creds = base64.b64encode(f"{TWILIO_ACCOUNT_SID}:{TWILIO_AUTH_TOKEN}".encode()).decode()
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Authorization", f"Basic {creds}")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            payload = json.loads(e.read())
        except Exception:
            payload = {"error": str(e)}
        return e.code, payload
    except Exception as e:
        return 502, {"error": str(e)}


def normalize_phone(raw: str) -> str | None:
    digits = re.sub(r"\D", "", raw or "")
    if len(digits) != 10:
        return None
    return f"+91{digits}"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/otp/send", methods=["POST"])
def otp_send():
    data = request.get_json(silent=True) or {}
    phone = normalize_phone(data.get("phone", ""))
    if not phone:
        return jsonify({"error": "Enter a valid 10-digit number"}), 400

    status, payload = twilio_post("Verifications", {"To": phone, "Channel": "sms"})
    if status >= 400:
        msg = payload.get("message") or payload.get("error") or "Failed to send OTP"
        return jsonify({"error": msg}), 502

    session["otp_phone"] = phone
    session["otp_verified"] = False
    return jsonify({"ok": True, "phone": phone, "status": payload.get("status")})


@app.route("/otp/verify", methods=["POST"])
def otp_verify():
    data = request.get_json(silent=True) or {}
    phone = session.get("otp_phone") or normalize_phone(data.get("phone", ""))
    code = (data.get("code") or "").strip()
    if not phone:
        return jsonify({"error": "No phone number on record. Resend OTP."}), 400
    if not re.fullmatch(r"\d{4,8}", code):
        return jsonify({"error": "Enter the OTP code"}), 400

    status, payload = twilio_post("VerificationCheck", {"To": phone, "Code": code})
    if status >= 400:
        msg = payload.get("message") or payload.get("error") or "Verification failed"
        return jsonify({"error": msg}), 502

    if payload.get("status") == "approved" and payload.get("valid"):
        session["otp_verified"] = True
        session["otp_phone"] = phone
        with _reporters_lock:
            stored_reporter = _reporters_data.get(phone)
        with _admins_lock:
            stored_admin = _admins_data.get(phone)
        with _partners_lock:
            stored_partner = _partners_data.get(phone)
        first_name = None
        if stored_reporter:
            session["reporter"] = stored_reporter
            first_name = (stored_reporter.get("full_name", "").split() or [""])[0]
        else:
            session.pop("reporter", None)
        if stored_admin:
            session["admin"] = stored_admin
            if not first_name:
                first_name = (stored_admin.get("full_name", "").split() or [""])[0]
        else:
            session.pop("admin", None)
        if stored_partner:
            stored_partner = _with_partner_experience(stored_partner, phone)
            session["partner"] = stored_partner
            if not first_name:
                first_name = (stored_partner.get("full_name", "").split() or [""])[0]
        else:
            session.pop("partner", None)
        return jsonify({
            "ok": True,
            "status": "approved",
            "phone": phone,
            "reporter": stored_reporter,
            "admin": stored_admin,
            "partner": stored_partner,
            "first_name": first_name,
        })

    return jsonify({"ok": False, "status": payload.get("status", "pending"), "error": "Incorrect OTP"}), 400


@app.route("/otp/test_verify", methods=["POST"])
def otp_test_verify():
    data = request.get_json(silent=True) or {}
    phone = normalize_phone(data.get("phone", ""))
    code = (data.get("code") or "").strip()
    if not phone:
        return jsonify({"error": "Enter a valid 10-digit number"}), 400
    if not re.fullmatch(r"\d+", code):
        return jsonify({"error": "Enter the OTP code"}), 400
    if code != "123456":
        return jsonify({"ok": False, "error": "Incorrect test OTP — only 123456 is accepted"}), 400

    session["otp_phone"] = phone
    session["otp_verified"] = True
    with _reporters_lock:
        stored_reporter = _reporters_data.get(phone)
    with _admins_lock:
        stored_admin = _admins_data.get(phone)
    with _partners_lock:
        stored_partner = _partners_data.get(phone)
    first_name = None
    if stored_reporter:
        session["reporter"] = stored_reporter
        first_name = (stored_reporter.get("full_name", "").split() or [""])[0]
    else:
        session.pop("reporter", None)
    if stored_admin:
        session["admin"] = stored_admin
        if not first_name:
            first_name = (stored_admin.get("full_name", "").split() or [""])[0]
    else:
        session.pop("admin", None)
    if stored_partner:
        stored_partner = _with_partner_experience(stored_partner, phone)
        session["partner"] = stored_partner
        if not first_name:
            first_name = (stored_partner.get("full_name", "").split() or [""])[0]
    else:
        session.pop("partner", None)
    return jsonify({
        "ok": True,
        "status": "approved",
        "phone": phone,
        "reporter": stored_reporter,
        "admin": stored_admin,
        "partner": stored_partner,
        "first_name": first_name,
    })


@app.route("/session")
def session_state():
    phone = session.get("otp_phone")
    reporter = session.get("reporter")
    admin = session.get("admin")
    partner = session.get("partner")
    # Backfill from persistent stores if the session got stale (e.g. user
    # logged in before profile-restore was wired into /otp/verify, or the
    # admin/reporter was created in another tab).
    if phone and session.get("otp_verified"):
        if not reporter:
            with _reporters_lock:
                stored_r = _reporters_data.get(phone)
            if stored_r:
                session["reporter"] = stored_r
                reporter = stored_r
        if not admin:
            with _admins_lock:
                stored_a = _admins_data.get(phone)
            if stored_a:
                session["admin"] = stored_a
                admin = stored_a
        if not partner:
            with _partners_lock:
                stored_p = _partners_data.get(phone)
            if stored_p:
                partner = _with_partner_experience(stored_p, phone)
                session["partner"] = partner
        elif partner:
            partner = _with_partner_experience(partner, phone)
            session["partner"] = partner
    first_name = None
    if reporter:
        first_name = (reporter.get("full_name", "").split() or [""])[0]
    elif admin:
        first_name = (admin.get("full_name", "").split() or [""])[0]
    elif partner:
        first_name = (partner.get("full_name", "").split() or [""])[0]
    return jsonify({
        "verified": bool(session.get("otp_verified")),
        "phone": phone,
        "reporter": reporter,
        "admin": admin,
        "partner": partner,
        "first_name": first_name,
        "total_points": _total_points_for(phone),
    })


EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
NAME_RE = re.compile(r"^[A-Za-z][A-Za-z\s.'-]{1,59}$")


@app.route("/signup", methods=["POST"])
def signup():
    if not session.get("otp_verified"):
        return jsonify({"error": "Phone not verified"}), 401
    data = request.get_json(silent=True) or {}
    full_name = (data.get("full_name") or "").strip()
    age_raw = data.get("age")
    email = (data.get("email") or "").strip()
    address = (data.get("address") or "").strip()
    home_zones = data.get("home_zones") or []

    errors = {}
    if not NAME_RE.fullmatch(full_name):
        errors["full_name"] = "Enter a valid full name"
    try:
        age = int(age_raw)
        if not (1 <= age <= 120):
            errors["age"] = "Age must be 1–120"
    except (TypeError, ValueError):
        errors["age"] = "Enter a valid age"
        age = None
    if not EMAIL_RE.match(email):
        errors["email"] = "Enter a valid email"
    if len(address) < 5:
        errors["address"] = "Enter a valid address"
    if errors:
        return jsonify({"errors": errors}), 400

    cleaned_zones = []
    seen = set()
    for z in home_zones:
        z = str(z).strip()[:60]
        if z and z.lower() not in seen:
            cleaned_zones.append(z)
            seen.add(z.lower())

    reporter = {
        "full_name": full_name,
        "age": age,
        "email": email,
        "address": address,
        "home_zones": cleaned_zones,
    }
    session["reporter"] = reporter
    phone = session.get("otp_phone")
    if phone:
        with _reporters_lock:
            _reporters_data[phone] = reporter
            _save_reporters()
    return jsonify({"ok": True, "first_name": full_name.split()[0]})


@app.route("/admin/signup", methods=["POST"])
def admin_signup():
    if not session.get("otp_verified"):
        return jsonify({"error": "Phone not verified"}), 401
    phone = session.get("otp_phone")
    data = request.get_json(silent=True) or {}
    full_name = (data.get("full_name") or "").strip()
    age_raw = data.get("age")
    email = (data.get("email") or "").strip()
    area = (data.get("area") or "").strip()
    categories = data.get("categories") or []

    errors = {}
    if not NAME_RE.fullmatch(full_name):
        errors["full_name"] = "Enter a valid full name"
    try:
        age = int(age_raw)
        if not (1 <= age <= 120):
            errors["age"] = "Age must be 1–120"
    except (TypeError, ValueError):
        errors["age"] = "Enter a valid age"
        age = None
    if not EMAIL_RE.match(email):
        errors["email"] = "Enter a valid email"
    if len(area) < 5:
        errors["area"] = "Enter the area under your concern"
    if not isinstance(categories, list) or not categories:
        errors["categories"] = "Pick at least one category"
    if errors:
        return jsonify({"errors": errors}), 400

    cleaned_cats = []
    seen = set()
    for c in categories:
        c = str(c).strip()[:60]
        if c and c.lower() not in seen:
            cleaned_cats.append(c)
            seen.add(c.lower())

    admin = {
        "full_name": full_name,
        "age": age,
        "email": email,
        "area": area,
        "categories": cleaned_cats,
    }
    session["admin"] = admin
    if phone:
        with _admins_lock:
            _admins_data[phone] = admin
            _save_admins()
    return jsonify({"ok": True, "first_name": full_name.split()[0], "admin": admin})


@app.route("/partner/admins")
def partner_admins():
    if not session.get("otp_verified"):
        return jsonify({"error": "Phone not verified"}), 401
    with _admins_lock:
        admins = [
            {
                "id": phone,
                "full_name": admin.get("full_name", ""),
                "area": admin.get("area", ""),
                "categories": admin.get("categories", []),
            }
            for phone, admin in _admins_data.items()
        ]
    admins.sort(key=lambda a: (a.get("full_name") or "").lower())
    return jsonify({"admins": admins})


@app.route("/partner/signup", methods=["POST"])
def partner_signup():
    if not session.get("otp_verified"):
        return jsonify({"error": "Phone not verified"}), 401
    phone = session.get("otp_phone")
    is_multipart = (request.content_type or "").lower().startswith("multipart/form-data")
    data = request.form if is_multipart else (request.get_json(silent=True) or {})
    full_name = (data.get("full_name") or "").strip()
    age_raw = data.get("age")
    experience_raw = data.get("experience_years")
    email = (data.get("email") or "").strip()
    raw_categories = data.get("categories") or data.get("trades") or []
    raw_availability = data.get("availability") or {}
    supervising_admin = (data.get("supervising_admin") or "").strip()
    if isinstance(raw_categories, str):
        try:
            categories = json.loads(raw_categories)
        except json.JSONDecodeError:
            categories = []
    else:
        categories = raw_categories
    if isinstance(raw_availability, str):
        try:
            availability = json.loads(raw_availability)
        except json.JSONDecodeError:
            availability = {}
    else:
        availability = raw_availability
    verification_docs = request.files.getlist("verification_docs") if is_multipart else []
    verification_docs = [f for f in verification_docs if f and f.filename]

    errors = {}
    if not NAME_RE.fullmatch(full_name):
        errors["full_name"] = "Enter a valid full name"
    try:
        age = int(age_raw)
        if not (1 <= age <= 120):
            errors["age"] = "Age must be 1–120"
    except (TypeError, ValueError):
        errors["age"] = "Enter a valid age"
        age = None
    try:
        experience_years = int(experience_raw)
        if not (0 <= experience_years <= 80):
            errors["experience_years"] = "Experience must be 0-80 years"
    except (TypeError, ValueError):
        errors["experience_years"] = "Enter years of experience"
        experience_years = None
    if not EMAIL_RE.match(email):
        errors["email"] = "Enter a valid email"
    if not isinstance(categories, list) or not categories:
        errors["categories"] = "Pick at least one category"
    weekdays = availability.get("weekdays") if isinstance(availability, dict) else None
    time_slots = availability.get("time_slots") if isinstance(availability, dict) else None
    legacy_time_slot = (availability.get("time_slot") or "").strip() if isinstance(availability, dict) else ""
    if not time_slots and legacy_time_slot:
        time_slots = [legacy_time_slot]
    if not isinstance(weekdays, list) or not weekdays:
        errors["availability"] = "Pick at least one weekday"
    if not isinstance(time_slots, list) or not time_slots:
        errors["time_slots"] = "Pick at least one time slot"
    if not supervising_admin:
        errors["supervising_admin"] = "Assign a supervising admin"
    doc_error = _validate_partner_docs(verification_docs)
    if doc_error:
        errors["verification_docs"] = doc_error
    if errors:
        return jsonify({"errors": errors}), 400

    cleaned_cats = []
    seen = set()
    for c in categories:
        c = str(c).strip()[:60]
        if c and c.lower() not in seen:
            cleaned_cats.append(c)
            seen.add(c.lower())

    valid_days = {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"}
    cleaned_days = []
    for d in weekdays:
        d = str(d).strip()[:3].title()
        if d in valid_days and d not in cleaned_days:
            cleaned_days.append(d)
    if not cleaned_days:
        return jsonify({"errors": {"availability": "Pick at least one weekday"}}), 400
    cleaned_slots = []
    seen_slots = set()
    for slot in time_slots:
        slot = str(slot).strip()[:80]
        if slot and slot.lower() not in seen_slots:
            cleaned_slots.append(slot)
            seen_slots.add(slot.lower())
    if not cleaned_slots:
        return jsonify({"errors": {"time_slots": "Pick at least one time slot"}}), 400

    with _admins_lock:
        admin = _admins_data.get(supervising_admin)
    if not admin:
        return jsonify({"errors": {"supervising_admin": "Pick a matching supervising admin"}}), 400
    admin_cats = {str(c).strip().lower() for c in admin.get("categories", [])}
    partner_cats = {c.lower() for c in cleaned_cats}
    if not (admin_cats & partner_cats):
        return jsonify({"errors": {"supervising_admin": "Pick an admin who handles at least one selected category"}}), 400

    created_at = time.time()
    saved_docs = _save_partner_docs(phone, verification_docs)
    partner = {
        "full_name": full_name,
        "age": age,
        "experience_years_at_signup": experience_years,
        "experience_years": experience_years,
        "created_at": created_at,
        "email": email,
        "categories": cleaned_cats,
        "availability": {
            "weekdays": cleaned_days,
            "time_slots": cleaned_slots,
        },
        "verification_documents": saved_docs,
        "supervising_admin": supervising_admin,
        "next_available_at": _snap_to_partner_slot(created_at, cleaned_days, cleaned_slots),
    }
    display_partner = _with_partner_experience(partner, phone)
    session["partner"] = display_partner
    if phone:
        with _partners_lock:
            _partners_data[phone] = partner
            _save_partners()
    return jsonify({"ok": True, "first_name": full_name.split()[0], "partner": display_partner})


@app.route("/partner/profile", methods=["POST"])
def partner_profile():
    if not session.get("otp_verified"):
        return jsonify({"error": "Phone not verified"}), 401
    phone = session.get("otp_phone")
    if not phone:
        return jsonify({"error": "No phone number on record"}), 400
    with _partners_lock:
        partner = dict(_partners_data.get(phone) or {})
    if not partner:
        partner = dict(session.get("partner") or {})
    if not partner:
        return jsonify({"error": "Complete service partner sign-up first"}), 400

    data = request.form
    raw_availability = data.get("availability") or {}
    supervising_admin = (data.get("supervising_admin") or "").strip()
    if isinstance(raw_availability, str):
        try:
            availability = json.loads(raw_availability)
        except json.JSONDecodeError:
            availability = {}
    else:
        availability = raw_availability
    verification_docs = [f for f in request.files.getlist("verification_docs") if f and f.filename]

    errors = {}
    weekdays = availability.get("weekdays") if isinstance(availability, dict) else None
    time_slots = availability.get("time_slots") if isinstance(availability, dict) else None
    if not isinstance(weekdays, list) or not weekdays:
        errors["availability"] = "Pick at least one weekday"
    if not isinstance(time_slots, list) or not time_slots:
        errors["time_slots"] = "Pick at least one time slot"
    if not supervising_admin:
        errors["supervising_admin"] = "Assign a supervising admin"
    existing_docs = partner.get("verification_documents") or []
    if existing_docs and verification_docs:
        errors["verification_docs"] = "Documents already uploaded"
    elif verification_docs:
        doc_error = _validate_partner_docs(verification_docs)
        if doc_error:
            errors["verification_docs"] = doc_error
    if errors:
        return jsonify({"errors": errors}), 400

    valid_days = {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"}
    cleaned_days = []
    for d in weekdays:
        d = str(d).strip()[:3].title()
        if d in valid_days and d not in cleaned_days:
            cleaned_days.append(d)
    if not cleaned_days:
        return jsonify({"errors": {"availability": "Pick at least one weekday"}}), 400
    cleaned_slots = []
    seen_slots = set()
    for slot in time_slots:
        slot = str(slot).strip()[:80]
        if slot and slot.lower() not in seen_slots:
            cleaned_slots.append(slot)
            seen_slots.add(slot.lower())
    if not cleaned_slots:
        return jsonify({"errors": {"time_slots": "Pick at least one time slot"}}), 400

    with _admins_lock:
        admin = _admins_data.get(supervising_admin)
    if not admin:
        return jsonify({"errors": {"supervising_admin": "Pick a matching supervising admin"}}), 400
    admin_cats = {str(c).strip().lower() for c in admin.get("categories", [])}
    partner_cats = {str(c).strip().lower() for c in partner.get("categories", [])}
    if not (admin_cats & partner_cats):
        return jsonify({"errors": {"supervising_admin": "Pick an admin who handles at least one selected category"}}), 400

    partner["availability"] = {"weekdays": cleaned_days, "time_slots": cleaned_slots}
    partner["supervising_admin"] = supervising_admin
    if not existing_docs and verification_docs:
        partner["verification_documents"] = _save_partner_docs(phone, verification_docs)

    display_partner = _with_partner_experience(partner, phone)
    session["partner"] = display_partner
    with _partners_lock:
        _partners_data[phone] = partner
        _save_partners()
    return jsonify({"ok": True, "partner": display_partner})


@app.route("/partner/docs/<path:stored_filename>")
def partner_doc(stored_filename):
    if not session.get("otp_verified"):
        return jsonify({"error": "Phone not verified"}), 401
    phone = session.get("otp_phone")
    if not phone:
        return jsonify({"error": "No phone number on record"}), 400
    safe_name = Path(stored_filename).name
    with _partners_lock:
        partner = _partners_data.get(phone) or {}
    allowed = {
        doc.get("stored_filename")
        for doc in (partner.get("verification_documents") or [])
        if doc.get("stored_filename")
    }
    if safe_name not in allowed:
        return jsonify({"error": "Document not found"}), 404
    safe_phone = re.sub(r"[^A-Za-z0-9_-]", "_", phone)
    return send_from_directory(UPLOAD_DIR / "partner_docs" / safe_phone, safe_name)


@app.route("/admin/reports")
def admin_list_all_reports():
    if not session.get("otp_verified"):
        return jsonify({"error": "Phone not verified"}), 401
    if not session.get("admin"):
        return jsonify({"error": "Admin only"}), 403
    admin = session.get("admin") or {}
    admin_area = admin.get("area") or ""
    area_filter = (request.args.get("area") or "").strip().lower()

    all_reports = []
    with _reports_lock:
        for ph, reports in _reports_data.items():
            for r in reports:
                if r.get("phone") is None:
                    r2 = dict(r); r2["phone"] = ph
                else:
                    r2 = r
                all_reports.append(_attach_reporter_info(r2))

    if area_filter in ("my", "other"):
        def matches_my(rep):
            for it in (rep.get("issues") or []):
                if _location_in_admin_area(it.get("location") or "", admin_area):
                    return True
            return False
        if area_filter == "my":
            all_reports = [r for r in all_reports if matches_my(r)]
        else:
            all_reports = [r for r in all_reports if not matches_my(r)]

    all_reports.sort(key=lambda r: r.get("created_at", 0), reverse=True)
    categories = sorted({
        (i.get("category") or "").strip()
        for r in all_reports for i in (r.get("issues") or [])
        if (i.get("category") or "").strip()
    })
    return jsonify({
        "reports": all_reports,
        "admin_area": admin_area,
        "categories": categories,
    })


@app.route("/admin/reports/<report_id>/seen", methods=["POST"])
def admin_mark_seen(report_id):
    if not session.get("admin"):
        return jsonify({"error": "Admin only"}), 403
    admin = session.get("admin") or {}
    _, report = _find_report(report_id)
    if not report:
        return jsonify({"error": "Report not found"}), 404
    if not report.get("seen_by_admin"):
        report["seen_by_admin"] = {
            "admin_name": admin.get("full_name", ""),
            "ts": time.time(),
        }
        with _reports_lock:
            _save_reports()
    return jsonify({"ok": True, "seen_by_admin": report.get("seen_by_admin")})


@app.route("/admin/reports/<report_id>/thumbs_up", methods=["POST"])
def admin_thumbs_up(report_id):
    if not session.get("admin"):
        return jsonify({"error": "Admin only"}), 403
    admin = session.get("admin") or {}
    _, report = _find_report(report_id)
    if not report:
        return jsonify({"error": "Report not found"}), 404
    if report.get("thumbs_up_by_admin"):
        return jsonify({"ok": True, "already": True,
                        "thumbs_up_by_admin": report["thumbs_up_by_admin"],
                        "total_points": report.get("points", 0)})
    BONUS = 50
    report["thumbs_up_by_admin"] = {
        "admin_name": admin.get("full_name", ""),
        "ts": time.time(),
        "points": BONUS,
    }
    report["points"] = int(report.get("points", 0)) + BONUS
    with _reports_lock:
        _save_reports()
    return jsonify({
        "ok": True,
        "thumbs_up_by_admin": report["thumbs_up_by_admin"],
        "total_points": report["points"],
    })


@app.route("/admin/reports/<report_id>/deny_issue", methods=["POST"])
def admin_deny_issue(report_id):
    if not session.get("admin"):
        return jsonify({"error": "Admin only"}), 403
    admin = session.get("admin") or {}
    data = request.get_json(silent=True) or {}
    issue_title = (data.get("issue_title") or "").strip()
    reason = (data.get("reason") or "").strip()
    if not issue_title:
        return jsonify({"error": "issue_title required"}), 400
    if not reason:
        return jsonify({"error": "Reason is required"}), 400
    _, report = _find_report(report_id)
    if not report:
        return jsonify({"error": "Report not found"}), 404
    denials = report.setdefault("denied_issues", [])
    existing = next((d for d in denials if d.get("issue_title") == issue_title), None)
    if existing:
        return jsonify({"ok": True, "already": True, "denied_issues": denials, "entry": existing})
    entry = {
        "issue_title": issue_title,
        "admin_name": admin.get("full_name", ""),
        "reason": reason[:500],
        "ts": time.time(),
    }
    denials.append(entry)
    with _reports_lock:
        _save_reports()
    return jsonify({"ok": True, "entry": entry, "denied_issues": denials})


def _ensure_assignments_dict(report):
    if not isinstance(report.get("assignments"), dict):
        report["assignments"] = {}
    return report["assignments"]


def _release_partner_slot(partner_phone, assignment):
    """Roll back a partner's next_available_at by the duration of an
    assignment that is being replaced. Best-effort — if their slot has
    moved further since, we keep that."""
    if not partner_phone or not assignment:
        return
    with _partners_lock:
        p = _partners_data.get(partner_phone)
        if not p:
            return
        try:
            duration_h = float(assignment.get("duration_hours") or 0)
        except (TypeError, ValueError):
            duration_h = 0
        try:
            scheduled = float(assignment.get("scheduled_for") or 0)
        except (TypeError, ValueError):
            scheduled = 0
        current_next = float(p.get("next_available_at") or 0)
        released_end = scheduled + duration_h * 3600
        # If the freed end was the latest commitment, roll back.
        if current_next and abs(current_next - released_end) < 60:
            p["next_available_at"] = scheduled
            _save_partners()


def _candidate_partners(issue_category):
    """All partners whose categories match the issue, sorted by earliest
    next_available_at."""
    candidates = []
    with _partners_lock:
        for ph, p in _partners_data.items():
            if not _category_match(issue_category, p.get("categories")):
                continue
            candidates.append((ph, dict(p)))
    candidates.sort(key=lambda x: float(x[1].get("next_available_at") or 0))
    return candidates


def _assign_partner(report, issue, partner_phone, partner, duration_hours, admin_name):
    """Atomically reserve the partner's next slot and write the assignment to
    the report. Returns the assignment dict."""
    with _partners_lock:
        live = _partners_data.get(partner_phone) or partner
        current_next = float(live.get("next_available_at") or time.time())
        weekdays = (live.get("availability") or {}).get("weekdays") or []
        time_slots = (live.get("availability") or {}).get("time_slots") or []
        start_floor = max(current_next, time.time())
        scheduled_for = _snap_to_partner_slot(start_floor, weekdays, time_slots)
        live["next_available_at"] = scheduled_for + duration_hours * 3600
        _partners_data[partner_phone] = live
        _save_partners()
    assignment = _assignment_dict(partner_phone, live, scheduled_for, duration_hours, admin_name)
    issue_title = issue.get("title") or issue.get("category") or "Issue"
    assignments = _ensure_assignments_dict(report)
    assignments[issue_title] = assignment
    return assignment


@app.route("/admin/reports/<report_id>/approve_issue", methods=["POST"])
def admin_approve_issue(report_id):
    if not session.get("admin"):
        return jsonify({"error": "Admin only"}), 403
    admin = session.get("admin") or {}
    data = request.get_json(silent=True) or {}
    issue_title = (data.get("issue_title") or "").strip()
    if not issue_title:
        return jsonify({"error": "issue_title required"}), 400
    _, report = _find_report(report_id)
    if not report:
        return jsonify({"error": "Report not found"}), 404
    issue = _find_issue_in_report(report, issue_title)
    if not issue:
        return jsonify({"error": "Issue not found in report"}), 404

    approved = report.setdefault("approved_issues", [])
    existing_approval = next((a for a in approved if a.get("issue_title") == issue_title), None)
    existing_assignment = (report.get("assignments") or {}).get(issue_title)
    if existing_approval and existing_assignment:
        return jsonify({
            "ok": True,
            "already": True,
            "assignment": existing_assignment,
            "approved_issues": approved,
        })

    duration_hours = _estimate_duration_hours(issue)
    candidates = _candidate_partners(issue.get("category") or "")
    no_partner = False
    assignment = None
    if not candidates:
        no_partner = True
    else:
        ph, partner = candidates[0]
        assignment = _assign_partner(
            report, issue, ph, partner, duration_hours, admin.get("full_name", "")
        )

    if not existing_approval:
        approved.append({
            "issue_title": issue_title,
            "admin_name": admin.get("full_name", ""),
            "admin_phone": session.get("otp_phone", ""),
            "ts": time.time(),
            "duration_hours": duration_hours,
        })

    if not report.get("timeline"):
        report["timeline"] = _build_timeline(float(report.get("created_at") or time.time()))
    _refresh_assigned_timeline(report)
    with _reports_lock:
        _save_reports()
    return jsonify({
        "ok": True,
        "no_partner": no_partner,
        "assignment": assignment,
        "approved_issues": approved,
        "assignments": report.get("assignments") or {},
        "timeline": report.get("timeline") or [],
        "duration_hours": duration_hours,
    })


@app.route("/admin/reports/<report_id>/change_assignment", methods=["POST"])
def admin_change_assignment(report_id):
    if not session.get("admin"):
        return jsonify({"error": "Admin only"}), 403
    admin = session.get("admin") or {}
    data = request.get_json(silent=True) or {}
    issue_title = (data.get("issue_title") or "").strip()
    new_partner_phone = (data.get("partner_phone") or "").strip()
    if not issue_title or not new_partner_phone:
        return jsonify({"error": "issue_title and partner_phone required"}), 400
    _, report = _find_report(report_id)
    if not report:
        return jsonify({"error": "Report not found"}), 404
    issue = _find_issue_in_report(report, issue_title)
    if not issue:
        return jsonify({"error": "Issue not found"}), 404
    assignments = _ensure_assignments_dict(report)
    prev = assignments.get(issue_title)
    if prev and prev.get("partner_phone") == new_partner_phone:
        return jsonify({"ok": True, "unchanged": True, "assignment": prev})
    with _partners_lock:
        new_partner = _partners_data.get(new_partner_phone)
    if not new_partner:
        return jsonify({"error": "Partner not found"}), 404
    if not _category_match(issue.get("category") or "", new_partner.get("categories")):
        return jsonify({"error": "Partner does not handle this category"}), 400

    duration_hours = (prev or {}).get("duration_hours") or _estimate_duration_hours(issue)
    try:
        duration_hours = float(duration_hours)
    except (TypeError, ValueError):
        duration_hours = 4

    if prev:
        _release_partner_slot(prev.get("partner_phone"), prev)

    assignment = _assign_partner(
        report, issue, new_partner_phone, new_partner, duration_hours,
        admin.get("full_name", ""),
    )
    if not report.get("timeline"):
        report["timeline"] = _build_timeline(float(report.get("created_at") or time.time()))
    _refresh_assigned_timeline(report)
    with _reports_lock:
        _save_reports()
    return jsonify({
        "ok": True,
        "assignment": assignment,
        "assignments": report.get("assignments") or {},
        "timeline": report.get("timeline") or [],
    })


@app.route("/admin/partners_for_issue")
def admin_partners_for_issue():
    if not session.get("admin"):
        return jsonify({"error": "Admin only"}), 403
    category = (request.args.get("category") or "").strip()
    q = (request.args.get("q") or "").strip().lower()
    rows = []
    with _partners_lock:
        items = list(_partners_data.items())
    for ph, p in items:
        if category and not _category_match(category, p.get("categories")):
            continue
        if q:
            blob = " ".join([
                p.get("full_name", ""),
                " ".join(p.get("categories") or []),
            ]).lower()
            if q not in blob:
                continue
        nxt = float(p.get("next_available_at") or p.get("created_at") or time.time())
        rows.append({
            "phone": ph,
            "full_name": p.get("full_name", ""),
            "categories": p.get("categories") or [],
            "next_available_at": nxt,
            "next_available_label": _next_available_label(nxt),
            "availability": p.get("availability") or {},
        })
    rows.sort(key=lambda r: r["next_available_at"])
    return jsonify({"partners": rows})


def _find_assignment_pair(report_id, issue_title, partner_phone):
    """Return (phone_key, report, issue, assignment) for the SP-owned
    assignment, or (None, None, None, None)."""
    with _reports_lock:
        for ph, reports in _reports_data.items():
            for r in reports:
                if r.get("id") != report_id:
                    continue
                assignments = r.get("assignments") or {}
                a = assignments.get(issue_title)
                if not a or a.get("partner_phone") != partner_phone:
                    return None, None, None, None
                issue = _find_issue_in_report(r, issue_title)
                return ph, r, issue, a
    return None, None, None, None


def _auto_approve_at(scheduled_for):
    try:
        return float(scheduled_for) - AUTO_APPROVE_LEAD_SECONDS
    except (TypeError, ValueError):
        return None


@app.route("/partner/assigned_reports")
def partner_assigned_reports():
    if not session.get("otp_verified") or not session.get("partner"):
        return jsonify({"error": "Service partner only"}), 403
    phone = session.get("otp_phone")
    out = []
    with _reports_lock:
        for ph, reports in _reports_data.items():
            for r in reports:
                assignments = r.get("assignments") or {}
                mine = {t: a for t, a in assignments.items() if a.get("partner_phone") == phone}
                if not mine:
                    continue
                # Build a slim report dict with only my-assignment context.
                slim = _attach_reporter_info(r)
                slim["my_assignments"] = mine
                out.append(slim)
    out.sort(key=lambda r: min(
        (float(a.get("scheduled_for") or 0) for a in (r.get("my_assignments") or {}).values()),
        default=0,
    ))
    return jsonify({"assignments": out})


def _apply_assignment_history(report, assignment, entry):
    history = assignment.setdefault("history", [])
    history.append(entry)
    _refresh_assigned_timeline(report)


def _set_report_status_from_assignments(report):
    """Roll up assignment statuses into the report-level status_label so
    list views convey the latest SP action."""
    assignments = (report.get("assignments") or {}).values()
    if not assignments:
        return
    accepted = [a for a in assignments if a.get("status") == "accepted"]
    denied = [a for a in assignments if a.get("status") == "denied"]
    rescheduled = [a for a in assignments if a.get("status") == "rescheduled"]
    if denied and not accepted:
        report["status_label"] = "SP denied. Task will be assigned to new SP"
    elif accepted:
        report["status_label"] = "Accepted by service partner"
    elif rescheduled:
        report["status_label"] = "Rescheduled by service partner"


@app.route("/partner/assignment/<report_id>/accept", methods=["POST"])
def partner_accept(report_id):
    if not session.get("partner"):
        return jsonify({"error": "Service partner only"}), 403
    phone = session.get("otp_phone")
    partner = session.get("partner") or {}
    data = request.get_json(silent=True) or {}
    issue_title = (data.get("issue_title") or "").strip()
    _, report, _, assignment = _find_assignment_pair(report_id, issue_title, phone)
    if not assignment:
        return jsonify({"error": "Assignment not found"}), 404
    if assignment.get("status") == "denied":
        return jsonify({"error": "Already denied"}), 400
    now = time.time()
    assignment["status"] = "accepted"
    assignment["accepted_at"] = now
    sp_name = partner.get("full_name", "")
    _apply_assignment_history(report, assignment, {
        "type": "accept", "sp_name": sp_name, "ts": now,
    })
    _set_report_status_from_assignments(report)
    with _reports_lock:
        _save_reports()
    return jsonify({"ok": True, "assignment": assignment, "status_label": report.get("status_label", "")})


@app.route("/partner/assignment/<report_id>/deny", methods=["POST"])
def partner_deny(report_id):
    if not session.get("partner"):
        return jsonify({"error": "Service partner only"}), 403
    phone = session.get("otp_phone")
    partner = session.get("partner") or {}
    data = request.get_json(silent=True) or {}
    issue_title = (data.get("issue_title") or "").strip()
    _, report, _, assignment = _find_assignment_pair(report_id, issue_title, phone)
    if not assignment:
        return jsonify({"error": "Assignment not found"}), 404
    now = time.time()
    assignment["status"] = "denied"
    assignment["denied_at"] = now
    assignment["auto_approve_at"] = _auto_approve_at(assignment.get("scheduled_for"))
    sp_name = partner.get("full_name", "")
    _apply_assignment_history(report, assignment, {
        "type": "deny", "sp_name": sp_name, "ts": now,
    })
    # Release SP's slot since they're walking away.
    _release_partner_slot(phone, assignment)
    _set_report_status_from_assignments(report)
    with _reports_lock:
        _save_reports()
    return jsonify({"ok": True, "assignment": assignment, "status_label": report.get("status_label", "")})


@app.route("/partner/assignment/<report_id>/reschedule", methods=["POST"])
def partner_reschedule(report_id):
    if not session.get("partner"):
        return jsonify({"error": "Service partner only"}), 403
    phone = session.get("otp_phone")
    partner = session.get("partner") or {}
    data = request.get_json(silent=True) or {}
    issue_title = (data.get("issue_title") or "").strip()
    date_str = (data.get("date") or "").strip()
    time_slot = (data.get("time_slot") or "").strip()
    if not date_str or not time_slot:
        return jsonify({"error": "date and time_slot required"}), 400
    slot_range = PARTNER_TIME_SLOT_RANGES.get(time_slot)
    if not slot_range:
        return jsonify({"error": "Invalid time slot"}), 400
    try:
        new_dt = datetime.strptime(date_str, "%Y-%m-%d").replace(hour=slot_range[0])
    except ValueError:
        return jsonify({"error": "Invalid date format (use YYYY-MM-DD)"}), 400
    new_epoch = new_dt.timestamp()
    if new_epoch <= time.time():
        return jsonify({"error": "Pick a future date/slot"}), 400

    _, report, _, assignment = _find_assignment_pair(report_id, issue_title, phone)
    if not assignment:
        return jsonify({"error": "Assignment not found"}), 404
    count = int(assignment.get("reschedule_count") or 0)
    if count >= MAX_RESCHEDULES:
        return jsonify({"error": "Reschedule limit reached"}), 400

    duration_hours = float(assignment.get("duration_hours") or 4)
    old_scheduled = assignment.get("scheduled_for")

    # Release old window; reserve new one at the end of the partner's queue
    # but no earlier than the requested slot start.
    with _partners_lock:
        live = _partners_data.get(phone) or {}
        try:
            current_next = float(live.get("next_available_at") or 0)
        except (TypeError, ValueError):
            current_next = 0
        # Roll back if this assignment was the latest commitment.
        old_end = float(old_scheduled or 0) + float(assignment.get("duration_hours") or 0) * 3600
        if current_next and abs(current_next - old_end) < 60:
            current_next = float(old_scheduled or current_next)
        scheduled_for = max(new_epoch, current_next, time.time())
        live["next_available_at"] = scheduled_for + duration_hours * 3600
        _partners_data[phone] = live
        _save_partners()

    sp_name = partner.get("full_name", "")
    now = time.time()
    assignment["scheduled_for"] = scheduled_for
    assignment["scheduled_for_label"] = _next_available_label(scheduled_for)
    assignment["reschedule_count"] = count + 1
    assignment["status"] = "rescheduled"
    # Per spec: a reschedule submit also counts as acceptance — so include
    # an accepted_at timestamp without flipping status to accepted (the UI
    # uses status == "rescheduled" to decide which buttons to hide).
    assignment["accepted_at"] = now
    assignment["auto_approve_at"] = None
    _apply_assignment_history(report, assignment, {
        "type": "reschedule",
        "sp_name": sp_name,
        "ts": now,
        "new_scheduled_for": scheduled_for,
        "new_scheduled_for_label": assignment["scheduled_for_label"],
        "time_slot": time_slot,
    })
    _set_report_status_from_assignments(report)
    with _reports_lock:
        _save_reports()
    return jsonify({"ok": True, "assignment": assignment, "status_label": report.get("status_label", "")})


TASK_MEDIA_MAX_TOTAL = 100 * 1024 * 1024  # 100 MB per phase
TASK_MEDIA_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".mp4", ".mov", ".webm", ".m4v"}


def _location_overlap(a, b, min_words=2):
    aw = _area_words(a)
    bw = _area_words(b)
    if not aw or not bw:
        return False
    return len(aw & bw) >= min_words


def _task_slug(s):
    return re.sub(r"[^A-Za-z0-9_-]", "_", s or "")[:60] or "task"


def _save_task_media(report_id, issue_title, phase, files):
    issue_slug = _task_slug(issue_title)
    dest_rel = Path("task_media") / report_id / issue_slug / phase
    dest_abs = FRAMES_DIR.parent / dest_rel  # static/task_media/...
    dest_abs.mkdir(parents=True, exist_ok=True)
    saved = []
    total = 0
    for f in files:
        if not f or not f.filename:
            continue
        ext = Path(f.filename).suffix.lower()
        if ext not in TASK_MEDIA_EXTS:
            return None, f"Unsupported file type: {f.filename}"
        size = _filestorage_size(f)
        total += size
        if total > TASK_MEDIA_MAX_TOTAL:
            return None, "Files exceed 100 MB total"
        safe_name = f"{uuid.uuid4().hex[:8]}_{re.sub(r'[^A-Za-z0-9._-]', '_', f.filename)[:80]}"
        out = dest_abs / safe_name
        f.save(out)
        kind = "video" if ext in {".mp4", ".mov", ".webm", ".m4v"} else "image"
        saved.append({
            "url": f"/static/{dest_rel.as_posix()}/{safe_name}",
            "filename": f.filename,
            "kind": kind,
            "size": size,
        })
    if not saved:
        return None, "Attach at least one file"
    return saved, ""


def _ai_evaluate_task(issue, before_media, after_media):
    """Use Gemini multimodal to score the fix on 1-5 stars and return a
    short comment. Best-effort — falls back to a neutral score if the
    call fails."""
    fallback = {"stars": 3, "comment": "Unable to auto-evaluate; please review manually."}
    try:
        parts = [
            "You are evaluating a residential-society task completion.\n"
            f"Issue title: {issue.get('title') or ''}\n"
            f"Issue category: {issue.get('category') or ''}\n"
            f"Issue detail: {issue.get('detail') or ''}\n\n"
            "Below are 'before' images (taken by the service partner before "
            "starting work) followed by 'after' images (taken after the work "
            "ended). Compare them and judge how well the issue was resolved.\n"
            "Return ONLY a JSON object: "
            '{"stars": <int 1-5>, "comment": "1-3 sentences"}. '
            "If the 'after' images appear to be from a different location "
            "than the 'before' images, explicitly call that out in the "
            "comment and lower the stars accordingly."
        ]

        def attach(label, items):
            chunks = [f"\n--- {label} ---"]
            for m in items[:6]:
                p = _url_to_path(m.get("url"))
                if not p or not p.exists():
                    continue
                if m.get("kind") == "image":
                    try:
                        with open(p, "rb") as fh:
                            chunks.append(types.Part.from_bytes(data=fh.read(), mime_type="image/jpeg"))
                    except Exception:
                        continue
            return chunks

        contents = [parts[0]] + attach("BEFORE", before_media) + attach("AFTER", after_media)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json",
            ),
        )
        data = parse_json_response(response.text)
        stars = int(data.get("stars") or 3)
        stars = max(1, min(5, stars))
        comment = str(data.get("comment") or "").strip()[:600] or fallback["comment"]
        return {"stars": stars, "comment": comment}
    except Exception:
        return fallback


@app.route("/partner/assignment/<report_id>/start_task", methods=["POST"])
def partner_start_task(report_id):
    if not session.get("partner"):
        return jsonify({"error": "Service partner only"}), 403
    phone = session.get("otp_phone")
    partner = session.get("partner") or {}
    issue_title = (request.form.get("issue_title") or "").strip()
    location = (request.form.get("location") or "").strip()
    if not issue_title:
        return jsonify({"error": "issue_title required"}), 400
    if len(location) < 5:
        return jsonify({"error": "Detect or enter your current location"}), 400
    confirm1 = request.form.get("confirm_society_permission") in ("1", "true", "on", "yes")
    confirm2 = request.form.get("confirm_admin_supervision") in ("1", "true", "on", "yes")
    if not (confirm1 and confirm2):
        return jsonify({"error": "Please tick both confirmation checkboxes"}), 400
    files = [f for f in request.files.getlist("before_media") if f and f.filename]
    if not files:
        return jsonify({"error": "Attach at least one 'before starting' file"}), 400

    _, report, issue, assignment = _find_assignment_pair(report_id, issue_title, phone)
    if not assignment:
        return jsonify({"error": "Assignment not found"}), 404
    if (assignment.get("status") or "").lower() not in {"accepted", "rescheduled"}:
        return jsonify({"error": "Accept the task before starting"}), 400
    if assignment.get("started_at"):
        return jsonify({"error": "Task already started"}), 400

    expected_loc = (issue or {}).get("location") or ""
    if expected_loc and not _location_overlap(expected_loc, location):
        return jsonify({"error": "Enter correct location of the issue (does not match)"}), 400

    saved, err = _save_task_media(report_id, issue_title, "before", files)
    if err:
        return jsonify({"error": err}), 400

    now = time.time()
    assignment["started_at"] = now
    assignment["started_location"] = location
    assignment["before_media"] = saved
    assignment["status"] = "in_progress"
    _apply_assignment_history(report, assignment, {
        "type": "start_task", "sp_name": partner.get("full_name", ""), "ts": now,
    })
    # advance timeline
    for step in report.get("timeline") or []:
        if step.get("step") == "started":
            step["done"] = True
            step["current"] = False
            step["ts"] = now
    with _reports_lock:
        _save_reports()
    return jsonify({"ok": True, "assignment": assignment, "status_label": report.get("status_label", "")})


@app.route("/partner/assignment/<report_id>/end_task", methods=["POST"])
def partner_end_task(report_id):
    if not session.get("partner"):
        return jsonify({"error": "Service partner only"}), 403
    phone = session.get("otp_phone")
    partner = session.get("partner") or {}
    issue_title = (request.form.get("issue_title") or "").strip()
    location = (request.form.get("location") or "").strip()
    if not issue_title:
        return jsonify({"error": "issue_title required"}), 400
    if len(location) < 5:
        return jsonify({"error": "Detect or enter your current location"}), 400
    confirm1 = request.form.get("confirm_admin_supervision_end") in ("1", "true", "on", "yes")
    confirm2 = request.form.get("confirm_task_complete") in ("1", "true", "on", "yes")
    if not (confirm1 and confirm2):
        return jsonify({"error": "Please tick both confirmation checkboxes"}), 400
    files = [f for f in request.files.getlist("after_media") if f and f.filename]
    if not files:
        return jsonify({"error": "Attach at least one 'after ending' file"}), 400

    _, report, issue, assignment = _find_assignment_pair(report_id, issue_title, phone)
    if not assignment:
        return jsonify({"error": "Assignment not found"}), 404
    if not assignment.get("started_at"):
        return jsonify({"error": "Start the task before ending"}), 400
    if assignment.get("ended_at"):
        return jsonify({"error": "Task already ended"}), 400

    expected_loc = (issue or {}).get("location") or assignment.get("started_location") or ""
    if expected_loc and not _location_overlap(expected_loc, location):
        return jsonify({"error": "Enter correct location of the issue (does not match)"}), 400

    saved, err = _save_task_media(report_id, issue_title, "after", files)
    if err:
        return jsonify({"error": err}), 400

    now = time.time()
    assignment["ended_at"] = now
    assignment["ended_location"] = location
    assignment["after_media"] = saved
    assignment["status"] = "completed"
    _apply_assignment_history(report, assignment, {
        "type": "end_task", "sp_name": partner.get("full_name", ""), "ts": now,
    })
    for step in report.get("timeline") or []:
        if step.get("step") == "completed":
            step["done"] = True
            step["current"] = False
            step["ts"] = now

    # Run AI evaluation synchronously — typical request finishes in 5-15s.
    evaluation = _ai_evaluate_task(issue or {}, assignment.get("before_media") or [], saved)
    evaluation["ts"] = now
    assignment["evaluation"] = evaluation

    with _reports_lock:
        _save_reports()
    return jsonify({
        "ok": True,
        "assignment": assignment,
        "evaluation": evaluation,
        "status_label": report.get("status_label", ""),
    })


def _auto_approve_denied_assignments():
    """Background worker: re-run the assign flow to pick a fresh SP for
    assignments that either (a) the SP explicitly denied, or (b) the SP
    didn't accept/reschedule/deny by 2 hours before the scheduled slot."""
    while True:
        try:
            now = time.time()
            todo = []
            with _reports_lock:
                for ph, reports in _reports_data.items():
                    for r in reports:
                        for title, a in (r.get("assignments") or {}).items():
                            status = (a.get("status") or "pending").lower()
                            if status == "denied":
                                t = a.get("auto_approve_at")
                                if t and float(t) <= now:
                                    todo.append((r, title, a, "denied"))
                                continue
                            if status == "pending":
                                # Untouched assignment: lapse it 2h before slot
                                sched = a.get("scheduled_for")
                                if sched and (float(sched) - AUTO_APPROVE_LEAD_SECONDS) <= now:
                                    todo.append((r, title, a, "lapsed"))
                                continue
            for r, title, a, reason in todo:
                if reason == "lapsed":
                    # Mark current assignment as denied due to inaction so
                    # downstream UI treats it identically to an SP-deny and
                    # the partner gets their slot back.
                    a["status"] = "denied"
                    a["denied_at"] = now
                    a["auto_lapsed"] = True
                    _release_partner_slot(a.get("partner_phone"), a)
                    history = a.setdefault("history", [])
                    history.append({
                        "type": "auto_lapse",
                        "sp_name": a.get("partner_name", ""),
                        "ts": now,
                        "reason": "SP did not respond in time",
                    })
                issue = _find_issue_in_report(r, title)
                if not issue:
                    continue
                category = issue.get("category") or ""
                duration_hours = float(a.get("duration_hours") or _estimate_duration_hours(issue))
                # Exclude the partner who denied this exact assignment.
                exclude_phone = a.get("partner_phone")
                candidates = [
                    (ph, p) for (ph, p) in _candidate_partners(category)
                    if ph != exclude_phone
                ]
                if not candidates:
                    a["auto_approve_attempted_at"] = now
                    a["auto_approve_failed"] = "No alternative service partner available"
                    continue
                ph, picked = candidates[0]
                assignment = _assign_partner(r, issue, ph, picked, duration_hours, "auto-reassign")
                assignment["auto_reassigned"] = True
                history = assignment.setdefault("history", [])
                history.append({
                    "type": "auto_reassign", "sp_name": picked.get("full_name", ""),
                    "ts": now, "reason": "previous SP denied; auto reassign 2h before slot",
                })
                if not r.get("timeline"):
                    r["timeline"] = _build_timeline(float(r.get("created_at") or time.time()))
                _refresh_assigned_timeline(r)
                _set_report_status_from_assignments(r)
            if todo:
                with _reports_lock:
                    _save_reports()
        except Exception:
            pass
        time.sleep(60)


threading.Thread(target=_auto_approve_denied_assignments, daemon=True).start()


@app.route("/admin/addresses")
def admin_addresses_lookup():
    if not session.get("otp_verified"):
        return jsonify({"error": "Phone not verified"}), 401
    q = (request.args.get("q") or "").strip().lower()
    addresses: set[str] = set()
    with _reports_lock:
        for _, reports in _reports_data.items():
            for r in reports:
                for it in (r.get("issues") or []):
                    loc = (it.get("location") or "").strip()
                    if loc:
                        addresses.add(loc)
    with _reporters_lock:
        for _, rep in _reporters_data.items():
            addr = (rep.get("address") or "").strip()
            if addr:
                addresses.add(addr)
    items = list(addresses)
    if q:
        items = [a for a in items if q in a.lower()]
    items.sort()
    return jsonify({"addresses": items[:10]})


@app.route("/profile", methods=["POST"])
def update_profile():
    if not session.get("otp_verified"):
        return jsonify({"error": "Phone not verified"}), 401
    reporter = session.get("reporter")
    phone = session.get("otp_phone")
    if not reporter or not phone:
        return jsonify({"error": "Not registered yet"}), 400

    data = request.get_json(silent=True) or {}
    address = (data.get("address") or "").strip()
    home_zones = data.get("home_zones") or []

    errors = {}
    if len(address) < 5:
        errors["address"] = "Enter a valid address"
    if errors:
        return jsonify({"errors": errors}), 400

    cleaned_zones = []
    seen = set()
    for z in home_zones:
        z = str(z).strip()[:60]
        if z and z.lower() not in seen:
            cleaned_zones.append(z)
            seen.add(z.lower())

    reporter["address"] = address
    reporter["home_zones"] = cleaned_zones
    session["reporter"] = reporter
    with _reporters_lock:
        _reporters_data[phone] = reporter
        _save_reporters()

    return jsonify({"ok": True, "reporter": reporter})


@app.route("/homezone/analyse", methods=["POST"])
def homezone_analyse():
    if not session.get("otp_verified"):
        return jsonify({"error": "Phone not verified"}), 401
    data = request.get_json(silent=True) or {}
    address = (data.get("address") or "").strip()
    if len(address) < 5:
        return jsonify({"error": "Enter the address first"}), 400

    prompt = (
        f'Address: "{address}"\n\n'
        "Suggest 3 short home-zone labels describing the kind of neighbourhood / property this likely is "
        '(e.g. "gated apartment complex", "urban high-rise", "suburban township", "row-house colony"). '
        'Return ONLY a JSON array of 3 short string labels, max 4 words each.'
    )
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt],
            config=types.GenerateContentConfig(
                temperature=0.3,
                response_mime_type="application/json",
            ),
        )
        zones = json.loads(response.text)
        if not isinstance(zones, list):
            zones = []
        zones = [str(z).strip()[:60] for z in zones if str(z).strip()][:5]
        return jsonify({"zones": zones})
    except Exception as e:
        return jsonify({"error": f"Gemini failed: {e}"}), 500


@app.route("/geocode/reverse", methods=["POST"])
def geocode_reverse():
    if not session.get("otp_verified"):
        return jsonify({"error": "Phone not verified"}), 401
    data = request.get_json(silent=True) or {}
    try:
        lat = float(data.get("lat"))
        lon = float(data.get("lon"))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid coordinates"}), 400

    url = (
        "https://nominatim.openstreetmap.org/reverse"
        f"?format=jsonv2&lat={lat}&lon={lon}&zoom=18&addressdetails=1"
    )
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "society-reporter-app/1.0 (sk39693@gmail.com)"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            payload = json.loads(resp.read())
    except Exception as e:
        return jsonify({"error": f"Reverse geocode failed: {e}"}), 502

    addr = payload.get("address") or {}
    parts = [
        addr.get("road") or addr.get("neighbourhood") or addr.get("suburb"),
        addr.get("suburb") if (addr.get("road") or addr.get("neighbourhood")) else None,
        addr.get("city") or addr.get("town") or addr.get("village") or addr.get("county"),
        addr.get("state"),
        addr.get("postcode"),
        addr.get("country"),
    ]
    short = ", ".join([p for p in parts if p])
    return jsonify({
        "address": short or payload.get("display_name") or "Unknown location",
        "display_name": payload.get("display_name", ""),
    })


def _run_describe_analyse(job_id: str, description: str, saved_files: list, tmp_dir: Path, job_frames_dir: Path):
    media_records = []
    try:
        for s in saved_files:
            try:
                ref = client.files.upload(file=str(s["path"]))
                while ref.state and ref.state.name == "PROCESSING":
                    time.sleep(2)
                    ref = client.files.get(name=ref.name)
                if ref.state and ref.state.name == "FAILED":
                    continue
                media_records.append({"tmp_path": s["path"], "is_video": s["is_video"], "ref": ref})
            except Exception:
                continue

        if not media_records:
            with _analysis_lock:
                _analysis_jobs[job_id].update({
                    "status": "failed",
                    "error": "Gemini could not process any of the uploaded files",
                })
            return

        n = len(media_records)
        index_lines = []
        for i, r in enumerate(media_records):
            if r["is_video"]:
                dur = video_duration(r["tmp_path"])
                index_lines.append(f"  index {i}: VIDEO, duration {dur:.2f} seconds")
            else:
                index_lines.append(f"  index {i}: IMAGE")
        index_desc = "\n".join(index_lines)

        prompt = (
            "You are reviewing media (images and/or videos) attached to a society / "
            "neighbourhood issue report.\n\n"
            f'Optional user description: "{description}"\n\n'
            "Attached media:\n"
            f"{index_desc}\n\n"
            "Identify each DISTINCT issue visible across the media. "
            f"Classify using ONLY these top-level categories:\n{ISSUE_CATEGORIES}\n\n"
            "Return ONLY valid JSON in this exact schema (numbers below are examples — replace them):\n"
            "{\n"
            '  "summary": "1-2 sentence overall summary",\n'
            '  "suggested_action": "1 sentence overall action",\n'
            '  "issues": [\n'
            "    {\n"
            '      "category": "top-level category name",\n'
            '      "title": "short issue title",\n'
            '      "detail": "what is visible and why it is an issue",\n'
            '      "severity": "low|medium|high",\n'
            '      "media_index": 1,\n'
            '      "start_time": 7.2,\n'
            '      "end_time": 11.8\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "CRITICAL rules:\n"
            f"- media_index MUST be an integer between 0 and {n - 1} (the media that best shows the issue).\n"
            "- For an IMAGE media, set start_time = 0 and end_time = 0.\n"
            "- For a VIDEO media, start_time and end_time MUST mark the SHORTEST window (in seconds) "
            "  that clearly captures the issue. Both values are decimal seconds from the start of "
            "  THAT video. Constraints:\n"
            "    * 0 <= start_time < end_time <= video duration shown above.\n"
            "    * The window should be 2 to 8 seconds long (extend slightly for context if needed).\n"
            "    * Different issues from the same video MUST have different windows that point to "
            "      where each specific issue is most visible.\n"
            "    * NEVER default to 0.0–0.0 for a video unless the issue is genuinely at the very start.\n"
            "- If no issues are visible, return an empty issues array."
        )
        contents = [r["ref"] for r in media_records] + [prompt]
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json",
            ),
        )
        result = parse_json_response(response.text)

        frames_dir = job_frames_dir
        frames_dir.mkdir(parents=True, exist_ok=True)
        issues = result.get("issues", []) or []

        def _f(v, default=0.0):
            try:
                return float(v)
            except (TypeError, ValueError):
                return default

        for i, issue in enumerate(issues):
            mi = issue.get("media_index", 0)
            try:
                mi = int(mi)
            except (TypeError, ValueError):
                mi = 0
            if mi < 0 or mi >= n:
                mi = 0
            rec = media_records[mi]
            issue["media_index"] = mi
            ok = False

            if rec["is_video"]:
                dur = video_duration(rec["tmp_path"])
                start = _f(issue.get("start_time"))
                end = _f(issue.get("end_time"))
                # sanity defaults if Gemini omitted or gave junk
                if dur > 0:
                    start = max(0.0, min(start, max(dur - 0.5, 0.0)))
                    if end <= start:
                        end = min(start + 4.0, dur)
                    end = max(start + 1.0, min(end, dur))
                    # cap clip length to 10s for sanity
                    if end - start > 10.0:
                        end = start + 10.0
                clip_name = f"issue_{i}.mp4"
                clip_path = frames_dir / clip_name
                ok = trim_video_clip(rec["tmp_path"], start, end, clip_path)
                if ok:
                    issue["media_kind"] = "video"
                    issue["media_url"] = f"/static/frames/report/{job_id}/{clip_name}"
                    issue["start_time"] = round(start, 2)
                    issue["end_time"] = round(end, 2)
                else:
                    # fallback: a still frame at start_time
                    jpg_name = f"issue_{i}.jpg"
                    if extract_frame(rec["tmp_path"], start, frames_dir / jpg_name):
                        issue["media_kind"] = "image"
                        issue["media_url"] = f"/static/frames/report/{job_id}/{jpg_name}"
                    else:
                        issue["media_kind"] = None
                        issue["media_url"] = None
            else:
                jpg_name = f"issue_{i}.jpg"
                out_path = frames_dir / jpg_name
                img = cv2.imread(str(rec["tmp_path"]))
                if img is not None:
                    h, w = img.shape[:2]
                    if max(h, w) > 1200:
                        scale = 1200 / max(h, w)
                        img = cv2.resize(img, (int(w * scale), int(h * scale)))
                    ok = cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
                issue["media_kind"] = "image" if ok else None
                issue["media_url"] = f"/static/frames/report/{job_id}/{jpg_name}" if ok else None
                if ok:
                    ann_name = f"issue_{i}_annotated.jpg"
                    ann_path = frames_dir / ann_name
                    if annotate_issue_image(out_path, issue.get("title") or "", issue.get("detail") or "", ann_path):
                        issue["annotated_url"] = f"/static/frames/report/{job_id}/{ann_name}"

        # Duplicate detection against previously submitted issues from all
        # reporters in the same or nearby location.
        with _analysis_lock:
            job_meta = _analysis_jobs.get(job_id) or {}
        phone = job_meta.get("phone")
        with _reports_lock:
            existing_reports = [
                r for reports in _reports_data.values() for r in reports
            ]
        # cache hash for previously-stored image-issues
        existing_records = []
        for r in existing_reports:
            reporter = _reporters_data.get(r.get("phone") or "", {})
            for prev in (r.get("issues") or []):
                prev_title = _issue_title(prev)
                text = ((prev.get("title") or "") + " " + (prev.get("detail") or "")).strip()
                loc = (prev.get("location") or "").strip()
                if prev.get("media_kind") == "image":
                    p = _url_to_path(prev.get("annotated_url") or prev.get("media_url"))
                    h = image_phash(p)
                else:
                    h = None
                avg_tol, user_tol, tol_count = _issue_tolerance_stats(r, prev_title, phone)
                existing_records.append({
                    "report_id": r.get("id"),
                    "issue_title": prev_title,
                    "owner_phone": r.get("phone"),
                    "reporter_name": reporter.get("full_name") or r.get("phone") or "the other user",
                    "upvotes": len(((r.get("issue_votes") or {}).get(prev_title) or [])),
                    "voted": phone in (((r.get("issue_votes") or {}).get(prev_title) or [])) if phone else False,
                    "avg_tolerance": avg_tol,
                    "user_tolerance": user_tol,
                    "tolerance_count": tol_count,
                    "text": text,
                    "location": loc,
                    "phash": h,
                    "kind": prev.get("media_kind"),
                })

        for i, issue in enumerate(issues):
            issue_text = ((issue.get("title") or "") + " " + (issue.get("detail") or "")).strip()
            issue_loc = (issue.get("location") or "").strip()
            new_path = _url_to_path(issue.get("annotated_url") or issue.get("media_url"))
            new_hash = image_phash(new_path) if issue.get("media_kind") == "image" else None
            dup = None
            for rec in existing_records:
                if issue_loc and rec["location"] and not _location_overlap(issue_loc, rec["location"], min_words=1):
                    continue
                if rec["text"] and text_jaccard(issue_text, rec["text"]) >= 0.55:
                    dup = rec
                    break
                if new_hash and rec["phash"] and hamming_distance(new_hash, rec["phash"]) <= 10:
                    dup = rec
                    break
            issue["source_issue_id"] = f"{job_id.upper()}-I{i + 1}"
            if dup:
                issue["is_duplicate"] = True
                issue["duplicate_of_report_id"] = dup["report_id"]
                issue["duplicate_issue_title"] = dup.get("issue_title")
                issue["duplicate_owner_phone"] = dup.get("owner_phone")
                issue["duplicate_reporter_name"] = dup.get("reporter_name") or "the other user"
                issue["duplicate_upvotes"] = dup.get("upvotes") or 0
                issue["duplicate_voted"] = bool(dup.get("voted"))
                issue["duplicate_avg_tolerance"] = dup.get("avg_tolerance")
                issue["duplicate_user_tolerance"] = dup.get("user_tolerance")
                issue["duplicate_tolerance_count"] = dup.get("tolerance_count") or 0
                issue["duplicate_can_interact"] = bool(phone and dup.get("owner_phone") != phone)
                issue["duplicate_can_add_tolerance"] = dup.get("user_tolerance") is None
            else:
                issue["is_duplicate"] = False

        result["issues"] = issues
        with _analysis_lock:
            _analysis_jobs[job_id].update({"status": "done", "result": result})
    except Exception as e:
        with _analysis_lock:
            _analysis_jobs[job_id].update({"status": "failed", "error": f"Gemini failed: {e}"})
    finally:
        for rec in media_records:
            try:
                client.files.delete(name=rec["ref"].name)
            except Exception:
                pass
            try:
                rec["tmp_path"].unlink()
            except Exception:
                pass
        try:
            tmp_dir.rmdir()
        except Exception:
            pass


@app.route("/describe/analyse", methods=["POST"])
def describe_analyse():
    if not session.get("otp_verified"):
        return jsonify({"error": "Phone not verified"}), 401
    phone = session.get("otp_phone")

    running_id = _get_running_job_for_phone(phone)
    if running_id:
        return jsonify({
            "error": "AI analysis is already running. Wait for completion.",
            "running_job_id": running_id,
        }), 409

    description = (request.form.get("description") or "").strip()
    evidence_files = request.files.getlist("evidence")
    evidence_files = [f for f in evidence_files if f and f.filename]
    if not evidence_files:
        return jsonify({"error": "Attach at least one media file"}), 400

    job_id = uuid.uuid4().hex[:12]
    tmp_dir = UPLOAD_DIR / "analyse" / job_id
    tmp_dir.mkdir(parents=True, exist_ok=True)
    job_frames_dir = FRAMES_DIR / "report" / job_id

    saved = []
    for f in evidence_files[:8]:
        safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", f.filename)[:120]
        tmp_path = tmp_dir / safe_name
        f.save(tmp_path)
        mime = (f.mimetype or "").lower()
        is_video = mime.startswith("video/") or tmp_path.suffix.lower() in {
            ".mp4", ".mov", ".webm", ".m4v", ".avi", ".mkv",
        }
        saved.append({"path": tmp_path, "is_video": is_video, "mime": mime})

    with _analysis_lock:
        _analysis_jobs[job_id] = {
            "status": "running",
            "phone": phone,
            "started_at": time.time(),
            "result": None,
            "error": None,
        }

    threading.Thread(
        target=_run_describe_analyse,
        args=(job_id, description, saved, tmp_dir, job_frames_dir),
        daemon=True,
    ).start()

    return jsonify({"job_id": job_id, "status": "running"})


@app.route("/describe/analyse/running")
def describe_analyse_running():
    if not session.get("otp_verified"):
        return jsonify({"running": False})
    phone = session.get("otp_phone")
    jid = _get_running_job_for_phone(phone)
    return jsonify({"running": bool(jid), "job_id": jid})


@app.route("/describe/analyse/status/<job_id>")
def describe_analyse_status(job_id):
    if not session.get("otp_verified"):
        return jsonify({"error": "Phone not verified"}), 401
    phone = session.get("otp_phone")
    with _analysis_lock:
        job = _analysis_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    if job.get("phone") != phone:
        return jsonify({"error": "Not authorized"}), 403
    return jsonify({
        "job_id": job_id,
        "status": job["status"],
        "result": job.get("result"),
        "error": job.get("error"),
    })


def _issues_context_text(issues: list) -> str:
    lines = []
    for it in issues[:10]:
        title = it.get("title") or it.get("category") or "Issue"
        cat = it.get("category") or "unknown"
        sev = it.get("severity") or "medium"
        line = f"- {title} ({cat}, severity: {sev})"
        detail = (it.get("detail") or "").strip()
        if detail:
            line += f" — {detail[:240]}"
        comment = (it.get("comment") or "").strip()
        if comment:
            line += f" | resident's note: \"{comment[:240]}\""
        loc = (it.get("location") or "").strip()
        if loc:
            line += f" | location: {loc[:120]}"
        lines.append(line)
    return "\n".join(lines)


@app.route("/chat/suggestions", methods=["POST"])
def chat_suggestions():
    if not session.get("otp_verified"):
        return jsonify({"error": "Phone not verified"}), 401
    data = request.get_json(silent=True) or {}
    issues = data.get("issues") or []
    if not isinstance(issues, list) or not issues:
        return jsonify({"suggestions": []})

    prompt = (
        "You help a resident who is reporting issues in their residential society (in India). "
        "Given the issues below, suggest 4 SHORT questions the resident might want to ask "
        "before submitting their report. Questions must be:\n"
        "- specific to these exact issues (and especially any resident notes if provided)\n"
        "- actionable / useful (next steps, responsibility, escalation, rights, timelines)\n"
        "- under 12 words each, no quotes, no numbering\n\n"
        f"Issues:\n{_issues_context_text(issues)}\n\n"
        "Return ONLY a JSON array of 4 question strings."
    )
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt],
            config=types.GenerateContentConfig(
                temperature=0.4,
                response_mime_type="application/json",
            ),
        )
        suggestions = json.loads(response.text)
        if not isinstance(suggestions, list):
            suggestions = []
        cleaned = []
        for s in suggestions:
            s = str(s).strip().strip('"').strip("'")
            if s:
                cleaned.append(s[:200])
        return jsonify({"suggestions": cleaned[:6]})
    except Exception as e:
        return jsonify({"error": f"Gemini failed: {e}"}), 500


@app.route("/chat/ask", methods=["POST"])
def chat_ask():
    if not session.get("otp_verified"):
        return jsonify({"error": "Phone not verified"}), 401
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Empty question"}), 400
    if len(question) > 1000:
        return jsonify({"error": "Question too long"}), 400
    issues = data.get("issues") or []
    history = data.get("history") or []

    convo = []
    for turn in history[-6:]:
        role = "User" if turn.get("role") == "user" else "Assistant"
        text = str(turn.get("text") or "").strip()
        if text:
            convo.append(f"{role}: {text}")
    convo.append(f"User: {question}\nAssistant:")
    convo_text = "\n".join(convo)

    prompt = (
        "You are a concise, practical assistant helping a resident of an Indian housing "
        "society / apartment complex. The resident is preparing a report about issues in "
        "their community. Answer the user's question in 3–5 short sentences. Be specific, "
        "actionable, and reference Indian RWA / society / municipal practice where relevant. "
        "Do NOT add disclaimers. Do NOT invent legal advice — point to the right body to ask.\n\n"
        f"Issues identified in the report:\n{_issues_context_text(issues) or '- (none)'}\n\n"
        "Conversation so far:\n"
        f"{convo_text}"
    )
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt],
            config=types.GenerateContentConfig(temperature=0.5),
        )
        return jsonify({"answer": (response.text or "").strip()})
    except Exception as e:
        return jsonify({"error": f"Gemini failed: {e}"}), 500


@app.route("/report", methods=["POST"])
def submit_report():
    if not session.get("otp_verified"):
        return jsonify({"error": "Phone not verified"}), 401
    reporter = session.get("reporter")
    if not reporter:
        return jsonify({"error": "Complete sign-up first"}), 400

    description = (request.form.get("description") or "").strip()
    evidence_files = request.files.getlist("evidence")
    analysis_job_id = (request.form.get("analysis_job_id") or "").strip()
    selected_issues_raw = request.form.get("selected_issues") or "[]"
    try:
        selected_issues = json.loads(selected_issues_raw)
        if not isinstance(selected_issues, list):
            selected_issues = []
    except json.JSONDecodeError:
        selected_issues = []

    report_id = uuid.uuid4().hex[:12]
    saved = []
    if evidence_files:
        evidence_dir = UPLOAD_DIR / "reports" / report_id
        evidence_dir.mkdir(parents=True, exist_ok=True)
        for f in evidence_files[:20]:
            if not f.filename:
                continue
            safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", f.filename)[:120]
            f.save(evidence_dir / safe_name)
            saved.append(safe_name)

    phone = session.get("otp_phone")
    running_job_id = _get_running_job_for_phone(phone)
    if running_job_id:
        return jsonify({"error": "AI analysis is still running. Please wait until it is complete."}), 409
    if not analysis_job_id:
        return jsonify({"error": "Run AI analysis before submitting this report."}), 400
    with _analysis_lock:
        job = _analysis_jobs.get(analysis_job_id)
    if not job or job.get("phone") != phone:
        return jsonify({"error": "Run AI analysis before submitting this report."}), 400
    if job.get("status") == "running":
        return jsonify({"error": "AI analysis is still running. Please wait until it is complete."}), 409
    if job.get("status") != "done":
        return jsonify({"error": job.get("error") or "AI analysis did not complete successfully."}), 400
    job_issues = ((job.get("result") or {}).get("issues") or [])
    if not job_issues:
        return jsonify({"error": "No issues were found in the AI analysis, so this report cannot be submitted."}), 400
    if not selected_issues:
        return jsonify({"error": "Select at least one non-duplicate issue before submitting."}), 400

    # Award points and persist the report.
    awarded = 0
    enriched_issues = []
    for it in selected_issues:
        pts = 0 if it.get("is_duplicate") else _points_for(it.get("severity"))
        awarded += pts
        enriched = dict(it)
        enriched["points"] = pts
        enriched_issues.append(enriched)

    now_ts = time.time()
    report_entry = {
        "id": report_id,
        "created_at": now_ts,
        "phone": phone,
        "description": description,
        "evidence_count": len(saved),
        "evidence_files": saved,
        "issues": enriched_issues,
        "points": awarded,
        "analysis_job_id": analysis_job_id,
        "status": "waiting_for_approval",
        "status_label": "Waiting for approval from admin",
        "timeline": _build_timeline(now_ts),
    }
    with _reports_lock:
        _reports_data.setdefault(phone, []).append(report_entry)
        _save_reports()

    return jsonify({
        "ok": True,
        "report_id": report_id,
        "reporter": reporter,
        "description": description,
        "evidence_count": len(saved),
        "issue_count": len(selected_issues),
        "selected_issues": enriched_issues,
        "points_awarded": awarded,
        "total_points": _total_points_for(phone),
        "analysis_running": False,
        "analysis_job_id": analysis_job_id,
    })


def _backfill_admin_phones(reports):
    """For reporter views: ensure approved_issues entries carry an
    admin_phone so the UI can render a tel: link. Looks up the admin by
    name from the admins store and patches in-place on the returned dict
    (the originals stay untouched)."""
    with _admins_lock:
        name_to_phone = {
            (a.get("full_name") or "").strip().lower(): ph
            for ph, a in _admins_data.items()
            if (a.get("full_name") or "").strip()
        }
    if not name_to_phone:
        return
    for r in reports:
        for a in r.get("approved_issues") or []:
            if a.get("admin_phone"):
                continue
            ph = name_to_phone.get((a.get("admin_name") or "").strip().lower())
            if ph:
                a["admin_phone"] = ph


def _issue_title(issue):
    return (issue or {}).get("title") or (issue or {}).get("category") or "Issue"


def _feed_category(issue):
    text = f"{(issue or {}).get('category') or ''} {_issue_title(issue)} {(issue or {}).get('detail') or ''}".lower()
    if any(w in text for w in ("waste", "garbage", "trash", "dump", "bin")):
        return "Waste Disposal"
    if any(w in text for w in ("water", "drain", "leak", "pipe", "sewage")):
        return "Water & Drainage"
    if any(w in text for w in ("light", "lamp", "streetlight", "street lighting")):
        return "Street Lighting"
    if any(w in text for w in ("road", "pothole", "footpath", "pavement", "traffic")):
        return "Roads & Potholes"
    if any(w in text for w in ("safety", "security", "guard", "cctv", "theft", "danger")):
        return "Public Safety"
    if any(w in text for w in ("park", "garden", "green", "bench", "tree")):
        return "Parks & Greenery"
    return "Infrastructure"


def _reporter_primary_location(phone):
    locations = []
    for r in _reports_data.get(phone, []):
        for issue in r.get("issues") or []:
            loc = (issue.get("location") or "").strip()
            if loc:
                locations.append(loc)
    if locations:
        scored = []
        for loc in locations:
            words = _area_words(loc)
            score = sum(len(words & _area_words(other)) for other in locations)
            scored.append((score, loc))
        scored.sort(reverse=True)
        return scored[0][1]
    rep = _reporters_data.get(phone) if phone else None
    return (rep or {}).get("address", "")


def _issue_feed_status(report, issue):
    title = _issue_title(issue)
    assignment = (report.get("assignments") or {}).get(title) or {}
    approved = next((a for a in report.get("approved_issues") or [] if a.get("issue_title") == title), None)
    denied = next((d for d in report.get("denied_issues") or [] if d.get("issue_title") == title), None)
    st = (assignment.get("status") or "").lower()
    if denied:
        return "Declined", bool(approved)
    if st == "completed" or assignment.get("ended_at"):
        return "Resolved", True
    if st == "in_progress" or assignment.get("started_at"):
        return "In Progress", True
    if st == "accepted":
        return "Acknowledged", True
    if assignment or approved:
        return "Acknowledged", True
    return "Reported", False


def _issue_vote_bucket(report, title):
    votes = report.setdefault("issue_votes", {})
    bucket = votes.setdefault(title, [])
    if not isinstance(bucket, list):
        bucket = list(bucket) if bucket else []
        votes[title] = bucket
    return bucket


def _issue_tolerance_bucket(report, title):
    tolerances = report.setdefault("issue_tolerances", {})
    bucket = tolerances.setdefault(title, [])
    if not isinstance(bucket, list):
        bucket = []
        tolerances[title] = bucket
    return bucket


def _issue_tolerance_stats(report, title, phone=None):
    bucket = ((report.get("issue_tolerances") or {}).get(title) or [])
    vals = []
    user_score = None
    for row in bucket:
        try:
            score = float(row.get("score"))
        except (TypeError, ValueError):
            continue
        score = max(0.0, min(100.0, score))
        vals.append(score)
        if phone and row.get("phone") == phone:
            user_score = score
    avg = round(sum(vals) / len(vals), 1) if vals else None
    return avg, user_score, len(vals)


def _feed_items_for(phone):
    base_loc = _reporter_primary_location(phone)
    rows = []
    seq = 1
    for owner_phone, reports in _reports_data.items():
        for report in reports:
            reporter = _reporters_data.get(report.get("phone") or owner_phone, {})
            for idx, issue in enumerate(report.get("issues") or []):
                title = _issue_title(issue)
                votes = ((report.get("issue_votes") or {}).get(title) or [])
                status, verified = _issue_feed_status(report, issue)
                loc = issue.get("location") or reporter.get("address", "")
                report_phone = report.get("phone") or owner_phone
                is_owner = bool(phone and report_phone == phone)
                avg_tol, user_tol, tol_count = _issue_tolerance_stats(report, title, phone)
                source_id = f"{(report.get('analysis_job_id') or report.get('id') or 'issue').upper()}-I{idx + 1}"
                row = {
                    "id": source_id,
                    "report_id": report.get("id", ""),
                    "issue_title": title,
                    "title": title,
                    "location": loc,
                    "category": issue.get("category") or "Uncategorized",
                    "filter_category": _feed_category(issue),
                    "priority": (issue.get("severity") or "medium").lower(),
                    "status": status,
                    "verified": verified,
                    "created_at": report.get("created_at") or 0,
                    "reporter_name": reporter.get("full_name") or report.get("phone") or "Reporter",
                    "media_count": int(report.get("evidence_count") or len(report.get("evidence_files") or [])),
                    "upvotes": max(0, len(votes)),
                    "voted": phone in votes if phone else False,
                    "is_owner": is_owner,
                    "can_upvote": not is_owner,
                    "avg_tolerance": avg_tol,
                    "tolerance_count": tol_count,
                    "user_tolerance": user_tol,
                    "can_add_tolerance": user_tol is None,
                    "near_score": len(_area_words(base_loc) & _area_words(loc)) if base_loc else 0,
                }
                rows.append(row)
                seq += 1
    nearby = [r for r in rows if r["near_score"] > 0]
    return nearby or rows


@app.route("/reports")
def list_reports():
    if not session.get("otp_verified"):
        return jsonify({"error": "Phone not verified"}), 401
    phone = session.get("otp_phone")
    with _reports_lock:
        reports = [_with_issue_statuses(r) for r in _reports_data.get(phone, [])]
    _backfill_admin_phones(reports)
    reports.sort(key=lambda r: r.get("created_at", 0), reverse=True)
    categories = sorted({
        (i.get("category") or "").strip()
        for r in reports for i in (r.get("issues") or [])
        if (i.get("category") or "").strip()
    })
    return jsonify({
        "reports": reports,
        "total_points": _total_points_for(phone),
        "categories": categories,
    })


@app.route("/explore/issues")
def explore_issues():
    if not session.get("otp_verified"):
        return jsonify({"error": "Phone not verified"}), 401
    phone = session.get("otp_phone")
    with _reports_lock:
        items = _feed_items_for(phone)
    items.sort(key=lambda x: x.get("created_at") or 0, reverse=True)
    return jsonify({
        "issues": items,
        "total_points": _total_points_for(phone),
        "categories": ADMIN_CATEGORIES,
    })


@app.route("/explore/issues/upvote", methods=["POST"])
def explore_issue_upvote():
    if not session.get("otp_verified"):
        return jsonify({"error": "Phone not verified"}), 401
    phone = session.get("otp_phone")
    data = request.get_json(silent=True) or {}
    report_id = (data.get("report_id") or "").strip()
    issue_title = (data.get("issue_title") or "").strip()
    if not report_id or not issue_title:
        return jsonify({"error": "report_id and issue_title required"}), 400
    _, report = _find_report(report_id)
    if not report:
        return jsonify({"error": "Issue not found"}), 404
    issue = _find_issue_in_report(report, issue_title)
    if not issue:
        return jsonify({"error": "Issue not found"}), 404
    if report.get("phone") == phone:
        return jsonify({"error": "You cannot upvote your own issue"}), 403
    votes = _issue_vote_bucket(report, issue_title)
    if phone in votes:
        votes[:] = [p for p in votes if p != phone]
        voted = False
    else:
        votes.append(phone)
        voted = True
    with _reports_lock:
        _save_reports()
    return jsonify({"ok": True, "upvotes": max(0, len(votes)), "voted": voted})


@app.route("/explore/issues/tolerance", methods=["POST"])
def explore_issue_tolerance():
    if not session.get("otp_verified"):
        return jsonify({"error": "Phone not verified"}), 401
    phone = session.get("otp_phone")
    data = request.get_json(silent=True) or {}
    report_id = (data.get("report_id") or "").strip()
    issue_title = (data.get("issue_title") or "").strip()
    try:
        score = float(data.get("score"))
    except (TypeError, ValueError):
        return jsonify({"error": "Enter a tolerance score from 0 to 100"}), 400
    if not (0 <= score <= 100):
        return jsonify({"error": "Tolerance score must be from 0 to 100"}), 400
    _, report = _find_report(report_id)
    if not report or not _find_issue_in_report(report, issue_title):
        return jsonify({"error": "Issue not found"}), 404
    bucket = _issue_tolerance_bucket(report, issue_title)
    if any(row.get("phone") == phone for row in bucket):
        avg, user_score, count = _issue_tolerance_stats(report, issue_title, phone)
        return jsonify({
            "error": "You already submitted a tolerance score for this issue",
            "avg_tolerance": avg,
            "user_tolerance": user_score,
            "tolerance_count": count,
        }), 400
    bucket.append({"phone": phone, "score": round(score, 1), "ts": time.time()})
    with _reports_lock:
        _save_reports()
    avg, user_score, count = _issue_tolerance_stats(report, issue_title, phone)
    return jsonify({
        "ok": True,
        "avg_tolerance": avg,
        "user_tolerance": user_score,
        "tolerance_count": count,
    })


@app.route("/rewards/catalog")
def rewards_catalog():
    if not session.get("otp_verified"):
        return jsonify({"error": "Phone not verified"}), 401
    phone = session.get("otp_phone")
    with _redemptions_lock:
        mine = list(_redemptions_data.get(phone, []))
    items = []
    for v in REWARDS_CATALOG:
        items.append({
            "id": v["id"],
            "name": v["name"],
            "brand": v["brand"],
            "amount_rs": v["amount_rs"],
            "points_cost": v["points_cost"],
        })
    mine_sorted = sorted(mine, key=lambda r: r.get("ts") or 0, reverse=True)
    return jsonify({
        "vouchers": items,
        "redeemed": mine_sorted,
        "total_points": _total_points_for(phone),
    })


@app.route("/rewards/redeem", methods=["POST"])
def rewards_redeem():
    if not session.get("otp_verified"):
        return jsonify({"error": "Phone not verified"}), 401
    phone = session.get("otp_phone")
    data = request.get_json(silent=True) or {}
    voucher_id = (data.get("voucher_id") or "").strip()
    voucher = next((v for v in REWARDS_CATALOG if v["id"] == voucher_id), None)
    if not voucher:
        return jsonify({"error": "Voucher not found"}), 404
    available = _total_points_for(phone)
    if available < voucher["points_cost"]:
        return jsonify({"error": f"You need {voucher['points_cost']} points to redeem this voucher. You have {available}."}), 400
    row = {
        "voucher_id": voucher["id"],
        "name": voucher["name"],
        "brand": voucher["brand"],
        "amount_rs": voucher["amount_rs"],
        "points_cost": voucher["points_cost"],
        "code": voucher["code"],
        "ts": time.time(),
    }
    with _redemptions_lock:
        _redemptions_data.setdefault(phone, []).append(row)
        _save_redemptions()
    return jsonify({
        "ok": True,
        "redeemed": row,
        "total_points": _total_points_for(phone),
    })


def _month_start_ts() -> float:
    now = datetime.now()
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return month_start.timestamp()


def _prev_month_window() -> tuple[float, float]:
    now = datetime.now()
    this_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    last_end = this_start
    last_start = (this_start - timedelta(days=1)).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return last_start.timestamp(), last_end.timestamp()


def _area_token(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "Unknown"
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        return text[:40]
    # Use the second part (locality/neighborhood) when present, else the first.
    if len(parts) >= 2:
        candidate = parts[1]
    else:
        candidate = parts[0]
    # Strip leading numbers like "123 ".
    candidate = re.sub(r"^\d+[\s,/-]+", "", candidate)
    return candidate[:40] or "Unknown"


@app.route("/explore/stats")
def explore_stats():
    if not session.get("otp_verified"):
        return jsonify({"error": "Phone not verified"}), 401
    month_start = _month_start_ts()
    prev_start, prev_end = _prev_month_window()
    cat_counts: dict = {c: 0 for c in ADMIN_CATEGORIES}
    cat_counts["Other"] = 0
    ward_rows: dict = {}
    total_issues = 0
    resolved_count = 0
    resolution_days_sum = 0.0
    resolution_days_n = 0
    prev_total = 0
    addresses_set = set()

    with _reports_lock:
        all_reports = []
        for owner_phone, reports in _reports_data.items():
            for r in reports:
                all_reports.append((owner_phone, r))

        for owner_phone, report in all_reports:
            created = float(report.get("created_at") or 0)
            issues = report.get("issues") or []
            for issue in issues:
                title = _issue_title(issue)
                loc = (issue.get("location") or "").strip()
                if loc:
                    addresses_set.add(loc.lower())
                if created < prev_end and created >= prev_start:
                    prev_total += 1
                if created < month_start:
                    continue
                total_issues += 1
                category = (issue.get("category") or "").strip()
                if category in cat_counts:
                    cat_counts[category] += 1
                else:
                    cat_counts["Other"] += 1
                area = _area_token(loc)
                if area not in ward_rows:
                    ward_rows[area] = {"area": area, "resolved": 0, "open": 0}
                assignment = (report.get("assignments") or {}).get(title) or {}
                status, _verified = _issue_feed_status(report, issue)
                ended_at = assignment.get("ended_at")
                started_at = assignment.get("started_at") or created
                if status == "Resolved" and ended_at:
                    resolved_count += 1
                    ward_rows[area]["resolved"] += 1
                    try:
                        days = max(0.0, (float(ended_at) - float(started_at)) / 86400.0)
                        resolution_days_sum += days
                        resolution_days_n += 1
                    except (TypeError, ValueError):
                        pass
                else:
                    ward_rows[area]["open"] += 1

    citizens_total = 0
    try:
        with _reporters_lock:
            citizens_total += len(_reporters_data)
        with _admins_lock:
            citizens_total += len(_admins_data)
        with _partners_lock:
            citizens_total += len(_partners_data)
    except NameError:
        pass

    resolution_rate = round(100.0 * resolved_count / total_issues, 1) if total_issues else 0.0
    avg_resolution_days = round(resolution_days_sum / resolution_days_n, 1) if resolution_days_n else 0.0
    delta_pct = None
    if prev_total:
        delta_pct = round(100.0 * (total_issues - prev_total) / prev_total, 1)
    elif total_issues:
        delta_pct = 100.0

    cats_payload = []
    for name in list(cat_counts.keys()):
        cats_payload.append({"name": name, "count": cat_counts[name]})

    wards_payload = []
    for row in ward_rows.values():
        total = row["resolved"] + row["open"]
        score = round(100.0 * row["resolved"] / total) if total else 0
        wards_payload.append({
            "area": row["area"],
            "resolved": row["resolved"],
            "open": row["open"],
            "score": score,
        })
    wards_payload.sort(key=lambda r: (-(r["resolved"] + r["open"]), r["area"]))
    wards_payload = wards_payload[:8]

    alert = None
    worst = None
    for row in ward_rows.values():
        total = row["resolved"] + row["open"]
        if total < 2:
            continue
        open_ratio = row["open"] / total if total else 0
        if worst is None or open_ratio > worst[0]:
            worst = (open_ratio, row)
    if worst:
        ratio, row = worst
        top_cat = max(cats_payload, key=lambda c: c["count"]) if cats_payload else None
        cat_name = (top_cat or {}).get("name") or "civic issues"
        confidence = min(95, 55 + int(ratio * 40))
        alert = {
            "title": "AI Predictive Alert",
            "body": (
                f"{row['area']} has seen a {row['open'] or 1}x backlog in {cat_name.lower()} reports this month — "
                "trend likely to continue without intervention. Recommended: proactive inspection and prioritised assignment."
            ),
            "confidence": confidence,
        }
    else:
        alert = {
            "title": "AI Predictive Alert",
            "body": "Not enough data this month to surface a confident prediction yet. Once more issues are reported, we'll flag emerging hotspots and recommended actions here.",
            "confidence": 0,
        }

    return jsonify({
        "total_reports": total_issues,
        "delta_pct": delta_pct,
        "resolution_rate": resolution_rate,
        "avg_resolution_days": avg_resolution_days,
        "citizens_engaged": citizens_total,
        "addresses_covered": len(addresses_set),
        "categories": cats_payload,
        "wards": wards_payload,
        "alert": alert,
        "month_label": datetime.now().strftime("%B %Y"),
    })


def _city_from_address(addr: str) -> str:
    parts = [p.strip() for p in (addr or "").split(",") if p.strip()]
    return parts[-1] if parts else ""


def _short_name(full: str) -> str:
    parts = (full or "").strip().split()
    if not parts:
        return "Reporter"
    if len(parts) == 1:
        return parts[0]
    return f"{parts[0]} {parts[-1][0].upper()}."


HERO_BADGES = [
    {
        "id": "first_reporter",
        "name": "First Reporter",
        "icon": "🚀",
        "desc": "Submit your first issue",
    },
    {
        "id": "lens_hero",
        "name": "Lens Hero",
        "icon": "📸",
        "desc": "Submit 5 photo reports",
    },
    {
        "id": "road_warrior",
        "name": "Road Warrior",
        "icon": "🛣",
        "desc": "Report 10 road issues",
    },
    {
        "id": "city_hero",
        "name": "City Hero",
        "icon": "🏆",
        "desc": "Top 3 in city ranking",
    },
]


def _badges_for(phone: str, reports: list, rank: int | None) -> list[str]:
    earned = []
    if reports:
        earned.append("first_reporter")
    photo_reports = sum(1 for r in reports if int(r.get("evidence_count") or 0) > 0 or (r.get("evidence_files") or []))
    if photo_reports >= 5:
        earned.append("lens_hero")
    road_count = 0
    for r in reports:
        for issue in r.get("issues") or []:
            cat = (issue.get("category") or "").lower()
            title = (issue.get("title") or "").lower()
            if "road" in cat or "pothole" in cat or "road" in title or "pothole" in title:
                road_count += 1
    if road_count >= 10:
        earned.append("road_warrior")
    if rank is not None and rank <= 3:
        earned.append("city_hero")
    return earned


@app.route("/explore/heroes")
def explore_heroes():
    if not session.get("otp_verified"):
        return jsonify({"error": "Phone not verified"}), 401
    me_phone = session.get("otp_phone")
    leaderboard = []
    with _reports_lock:
        all_data = {p: list(rs) for p, rs in _reports_data.items()}
    with _reporters_lock:
        reporters_snapshot = dict(_reporters_data)
    for phone, reports in all_data.items():
        if not reports:
            continue
        points = sum(int(r.get("points", 0)) for r in reports)
        if points <= 0:
            continue
        reporter = reporters_snapshot.get(phone, {})
        full = reporter.get("full_name") or "Reporter"
        # primary location for this contributor
        locs = []
        for r in reports:
            for issue in r.get("issues") or []:
                loc = (issue.get("location") or "").strip()
                if loc:
                    locs.append(loc)
        if not locs and reporter.get("address"):
            locs.append(reporter["address"])
        primary_loc = ""
        if locs:
            scored = sorted(((sum(len(_area_words(loc) & _area_words(o)) for o in locs), loc) for loc in locs), reverse=True)
            primary_loc = scored[0][1]
        leaderboard.append({
            "phone": phone,
            "name": _short_name(full),
            "full_name": full,
            "area": _area_token(primary_loc),
            "city": _city_from_address(primary_loc) or _city_from_address(reporter.get("address", "")),
            "reports": len(reports),
            "points": points,
        })
    leaderboard.sort(key=lambda r: (-r["points"], r["name"]))
    for i, row in enumerate(leaderboard):
        row["rank"] = i + 1

    my_row = next((r for r in leaderboard if r["phone"] == me_phone), None)
    me_reports = all_data.get(me_phone, [])
    me_reporter = reporters_snapshot.get(me_phone, {})
    me_full = me_reporter.get("full_name") or "You"
    me_locs = []
    for r in me_reports:
        for issue in r.get("issues") or []:
            loc = (issue.get("location") or "").strip()
            if loc:
                me_locs.append(loc)
    if not me_locs and me_reporter.get("address"):
        me_locs.append(me_reporter["address"])
    me_primary = ""
    if me_locs:
        scored = sorted(((sum(len(_area_words(loc) & _area_words(o)) for o in me_locs), loc) for loc in me_locs), reverse=True)
        me_primary = scored[0][1]
    me_points = sum(int(r.get("points", 0)) for r in me_reports)
    me_rank = my_row["rank"] if my_row else None
    me_badges = _badges_for(me_phone, me_reports, me_rank)
    profile = {
        "name": _short_name(me_full),
        "full_name": me_full,
        "area": _area_token(me_primary),
        "city": _city_from_address(me_primary) or _city_from_address(me_reporter.get("address", "")),
        "points": me_points,
        "reports": len(me_reports),
        "rank": me_rank,
        "scope": "City",
        "earned_badges": me_badges,
    }

    # Strip phone before returning leaderboard
    public_board = [
        {k: v for k, v in row.items() if k != "phone"}
        for row in leaderboard[:25]
    ]
    badges_payload = [
        {**b, "earned": b["id"] in me_badges}
        for b in HERO_BADGES
    ]
    return jsonify({
        "profile": profile,
        "leaderboard": public_board,
        "badges": badges_payload,
    })


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"ok": True})


@app.route("/static/frames/<path:filename>")
def serve_frame(filename):
    return send_from_directory(FRAMES_DIR, filename)


@app.route("/analyze", methods=["POST"])
def analyze():
    if not session.get("otp_verified"):
        return jsonify({"error": "Phone number not verified"}), 401
    if "video" not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400

    file = request.files["video"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    job_id = uuid.uuid4().hex[:12]
    ext = Path(file.filename).suffix or ".mp4"
    video_path = UPLOAD_DIR / f"{job_id}{ext}"
    file.save(video_path)

    try:
        result = analyze_video_with_gemini(video_path)
    except Exception as e:
        return jsonify({"error": f"Gemini analysis failed: {e}"}), 500

    job_frame_dir = FRAMES_DIR / job_id
    job_frame_dir.mkdir(parents=True, exist_ok=True)

    locations = result.get("locations", []) or []
    for idx, loc in enumerate(locations):
        ts = float(loc.get("timestamp", 0) or 0)
        frame_name = f"loc_{idx}.jpg"
        out_path = job_frame_dir / frame_name
        if extract_frame(video_path, ts, out_path):
            loc["image_url"] = f"/static/frames/{job_id}/{frame_name}"
        else:
            loc["image_url"] = None
        loc["timestamp"] = ts

    try:
        video_path.unlink()
    except Exception:
        pass

    return jsonify({
        "job_id": job_id,
        "summary": result.get("summary", ""),
        "locations": locations,
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5053")),
        debug=os.environ.get("FLASK_DEBUG") == "1",
    )
