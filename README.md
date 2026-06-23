# Society Video Inspector

Flask app that takes a video walkthrough of a housing society, sends it to Gemini for analysis, and shows the detected locations, best representative frame for each, and any flagged maintenance / security / waste / governance issues.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Then open http://localhost:5050.

The Gemini API key is baked into `app.py` (and can be overridden by setting the `GEMINI_API_KEY` env var).

## How it works

1. Browser uploads the video to `/analyze`.
2. The video is sent to Gemini (`gemini-2.0-flash`) via the Files API.
3. Gemini returns JSON with each location, the best timestamp, a description, and any issues.
4. The backend extracts a frame at each timestamp with OpenCV and serves it from `static/frames/`.
5. The frontend renders one card per location with the image and color-coded issue badges (high / medium / low).
