import os
import json
import time
import uuid
import re
import base64
import subprocess
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path

import cv2
from flask import Flask, request, jsonify, render_template, send_from_directory, session
from google import genai
from google.genai import types

GEMINI_API_KEY = os.environ.get(
    "GEMINI_API_KEY",
    "REDACTED_GEMINI_KEY",
)

# Image generation / editing model (Nano Banana 2 → fallback to Nano Banana).
# Override with IMAGE_GEN_MODELS env, e.g. "gemini-3-pro-image-preview,gemini-2.5-flash-image"
IMAGE_GEN_MODELS = os.environ.get(
    "IMAGE_GEN_MODELS",
    "gemini-3-pro-image-preview,gemini-2.5-flash-image,gemini-2.5-flash-image-preview",
).split(",")

TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "REDACTED_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "REDACTED_TOKEN")
TWILIO_SERVICE_SID = os.environ.get("TWILIO_SERVICE_SID", "REDACTED_SERVICE")

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
FRAMES_DIR = BASE_DIR / "static" / "frames"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB
app.secret_key = os.environ.get("FLASK_SECRET", "society-inspector-dev-secret-change-me")

client = genai.Client(api_key=GEMINI_API_KEY)

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
        return jsonify({"ok": True, "status": "approved", "phone": phone})

    return jsonify({"ok": False, "status": payload.get("status", "pending"), "error": "Incorrect OTP"}), 400


@app.route("/session")
def session_state():
    reporter = session.get("reporter")
    return jsonify({
        "verified": bool(session.get("otp_verified")),
        "phone": session.get("otp_phone"),
        "reporter": reporter,
        "first_name": (reporter.get("full_name", "").split() or [""])[0] if reporter else None,
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

    session["reporter"] = {
        "full_name": full_name,
        "age": age,
        "email": email,
        "address": address,
        "home_zones": cleaned_zones,
    }
    return jsonify({"ok": True, "first_name": full_name.split()[0]})


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


@app.route("/describe/analyse", methods=["POST"])
def describe_analyse():
    if not session.get("otp_verified"):
        return jsonify({"error": "Phone not verified"}), 401

    description = (request.form.get("description") or "").strip()
    evidence_files = request.files.getlist("evidence")
    evidence_files = [f for f in evidence_files if f and f.filename]
    if not evidence_files:
        return jsonify({"error": "Attach at least one media file"}), 400

    job_id = uuid.uuid4().hex[:12]
    tmp_dir = UPLOAD_DIR / "analyse" / job_id
    tmp_dir.mkdir(parents=True, exist_ok=True)
    media_records = []  # [{tmp_path, is_video, ref}]
    try:
        for f in evidence_files[:8]:
            safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", f.filename)[:120]
            tmp_path = tmp_dir / safe_name
            f.save(tmp_path)
            mime = (f.mimetype or "").lower()
            is_video = mime.startswith("video/") or tmp_path.suffix.lower() in {".mp4", ".mov", ".webm", ".m4v", ".avi", ".mkv"}
            try:
                ref = client.files.upload(file=str(tmp_path))
                while ref.state and ref.state.name == "PROCESSING":
                    time.sleep(2)
                    ref = client.files.get(name=ref.name)
                if ref.state and ref.state.name == "FAILED":
                    continue
                media_records.append({"tmp_path": tmp_path, "is_video": is_video, "ref": ref})
            except Exception:
                continue

        if not media_records:
            return jsonify({"error": "Gemini could not process any of the uploaded files"}), 500

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

        frames_dir = FRAMES_DIR / "report" / job_id
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

        result["issues"] = issues
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Gemini failed: {e}"}), 500
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


@app.route("/report", methods=["POST"])
def submit_report():
    if not session.get("otp_verified"):
        return jsonify({"error": "Phone not verified"}), 401
    reporter = session.get("reporter")
    if not reporter:
        return jsonify({"error": "Complete sign-up first"}), 400

    description = (request.form.get("description") or "").strip()
    evidence_files = request.files.getlist("evidence")
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

    return jsonify({
        "ok": True,
        "report_id": report_id,
        "reporter": reporter,
        "description": description,
        "evidence_count": len(saved),
        "issue_count": len(selected_issues),
        "selected_issues": selected_issues,
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
    app.run(host="0.0.0.0", port=5050, debug=True)
