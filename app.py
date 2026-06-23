import os
import json
import time
import uuid
import re
from pathlib import Path

import cv2
from flask import Flask, request, jsonify, render_template, send_from_directory
from google import genai
from google.genai import types

GEMINI_API_KEY = os.environ.get(
    "GEMINI_API_KEY",
    "REDACTED_GEMINI_KEY",
)

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
FRAMES_DIR = BASE_DIR / "static" / "frames"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB

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
    target_frame = min(int(timestamp_sec * fps), max(int(total_frames) - 1, 0))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return False
    cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return True


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


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/static/frames/<path:filename>")
def serve_frame(filename):
    return send_from_directory(FRAMES_DIR, filename)


@app.route("/analyze", methods=["POST"])
def analyze():
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
