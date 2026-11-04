"""
poster.py - posts the next unposted image from ./images to Instagram,
with an AI-written caption. Runs inside GitHub Actions (see .github/workflows/post.yml).

Secrets (set in GitHub, never in files):
    IG_ACCESS_TOKEN, IG_USER_ID, GEMINI_API_KEY
Optional environment variables:
    DRY_RUN=true      write the caption and print it, but do NOT post
    MAX_POSTS=1       how many images to post per run (default 1)
    GEMINI_MODEL=...  force a specific Gemini model name
    ALLOW_FALLBACK=true  post with a generic caption if Gemini is down (default: skip the post)
"""
import base64
import csv
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import requests

GRAPH = "https://graph.facebook.com/v26.0"
GEMINI = "https://generativelanguage.googleapis.com/v1beta/models"
IMAGES_DIR = Path("images")
POSTED_CSV = Path("posted.csv")
IMAGE_EXTS = {".jpg", ".jpeg"}
MODEL_CANDIDATES = [
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
    "gemini-flash-latest",
]

PROMPT = """You write Instagram captions for a men's hairstyle inspiration account.
Look at the image and write ONE caption in English:
- Line 1: a short hook that names the hairstyle(s) shown. Use style names printed in the image if there are any; otherwise give your best description, for example "Textured crop with a low fade".
- Then 2-3 short sentences: what makes the style work, who it suits (hair type or face shape), and one practical styling or barber tip.
- Finish with a short question that invites followers to comment or save the post.
- Then a blank line and 6-8 relevant hashtags.
Rules: plain text only, no markdown, at most 2 emojis, do not mention Pinterest, other websites or other creators, do not invent product names or brands, keep it under 900 characters in total.
Reply with the caption only."""

FALLBACK_CAPTION = (
    "Fresh men's haircut inspiration. Save this for your next barber visit "
    "and tell us what you would change.\n\n"
    "#menshair #mensHaircut #hairstyle #barber #haircutinspo #fade #menshairstyles"
)


def log(msg):
    print(msg, flush=True)


def get_posted():
    if not POSTED_CSV.exists():
        return set()
    with open(POSTED_CSV, encoding="utf-8", newline="") as f:
        return {row["file"] for row in csv.DictReader(f)}


def next_images(n):
    done = get_posted()
    files = sorted(p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    return [p for p in files if p.name not in done][:n]


def record(file_name, media_id, caption):
    new = not POSTED_CSV.exists()
    with open(POSTED_CSV, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["file", "posted_at_utc", "media_id", "caption"])
        w.writerow([file_name, datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    media_id, caption.replace("\n", " | ")])


def raw_url(file_name):
    repo = os.environ.get("GITHUB_REPOSITORY", "mrkevin453/ig-test-images-new")
    branch = os.environ.get("POST_BRANCH") or os.environ.get("GITHUB_REF_NAME") or "main"
    return f"https://raw.githubusercontent.com/{repo}/{branch}/images/{quote(file_name)}"


def wait_for_url(url):
    """Make sure the public link really returns an image before asking Instagram."""
    for attempt in range(6):
        try:
            r = requests.get(url, timeout=30, stream=True)
            ctype = r.headers.get("content-type", "")
            r.close()
            if r.status_code == 200 and ctype.startswith("image/"):
                return True
            log(f"  image link not ready (status {r.status_code}, {ctype}), retrying...")
        except requests.RequestException as e:
            log(f"  image link check failed: {e}")
        time.sleep(5)
    return False


def clean_caption(text):
    text = text.strip().replace("**", "").replace("```", "")
    tags = [w for w in text.split() if w.startswith("#")]
    if len(tags) > 30:  # Instagram allows at most 30 hashtags
        for extra in tags[30:]:
            text = text.replace(extra, "")
    return text.strip()[:2000]


def discover_models(key):
    """Ask Google which text+image models this key can use (names change over time)."""
    try:
        r = requests.get(GEMINI, headers={"x-goog-api-key": key},
                         params={"pageSize": 200}, timeout=30)
        items = r.json().get("models", []) if r.status_code == 200 else []
    except (requests.RequestException, ValueError):
        return []
    skip = ("image", "live", "audio", "tts", "embedding", "native", "robotics", "computer")
    names = []
    for m in items:
        name = m.get("name", "").split("/")[-1]
        if ("generateContent" in m.get("supportedGenerationMethods", [])
                and "flash" in name and not any(x in name for x in skip)):
            names.append(name)
    return sorted(names, reverse=True)


def try_models(models, key, body):
    for model in models:
        for attempt in range(3):
            try:
                r = requests.post(f"{GEMINI}/{model}:generateContent",
                                  headers={"x-goog-api-key": key}, json=body, timeout=90)
            except requests.RequestException as e:
                log(f"  Gemini request failed ({model}): {e}")
                time.sleep(5)
                continue
            if r.status_code in (403, 404):
                log(f"  model {model} not available to this key ({r.status_code}), trying the next one")
                break
            if r.status_code in (429, 500, 503):
                log(f"  Gemini busy ({r.status_code}) on {model}, waiting...")
                time.sleep(10 * (attempt + 1))
                continue
            if r.status_code != 200:
                log(f"  Gemini error {r.status_code} on {model}: {r.text[:300]}")
                break
            try:
                parts = r.json()["candidates"][0]["content"]["parts"]
                text = "".join(p.get("text", "") for p in parts).strip()
            except (KeyError, IndexError, ValueError):
                log("  Gemini returned no caption (possibly blocked), trying again")
                continue
            if len(text) > 30:
                log(f"  caption written by {model}")
                return clean_caption(text)
    return None


def make_caption(path):
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        log("  no GEMINI_API_KEY, using fallback caption")
        return FALLBACK_CAPTION
    data = base64.b64encode(path.read_bytes()).decode()
    body = {
        "contents": [{"parts": [
            {"inline_data": {"mime_type": "image/jpeg", "data": data}},
            {"text": PROMPT},
        ]}],
        "generationConfig": {"temperature": 0.8, "maxOutputTokens": 4000},
    }
    first = [os.environ["GEMINI_MODEL"]] if os.environ.get("GEMINI_MODEL") else MODEL_CANDIDATES
    caption = try_models(first, key, body)
    if caption:
        return caption
    log("  none of the usual models worked, asking Google which models this key can use...")
    found = [m for m in discover_models(key) if m not in first]
    log(f"  models found: {', '.join(found) if found else 'none'}")
    caption = try_models(found[:6], key, body)
    if caption:
        return caption
    log("  WARNING: could not get an AI caption, using the fallback caption")
    return FALLBACK_CAPTION


def ig_call(method, url, **params):
    params["access_token"] = os.environ["IG_ACCESS_TOKEN"]
    r = requests.request(method, url, params=params if method == "GET" else None,
                         data=params if method == "POST" else None, timeout=60)
    try:
        data = r.json()
    except ValueError:
        sys.exit(f"Unexpected response from Instagram ({r.status_code}): {r.text[:300]}")
    if "error" in data:
        e = data["error"]
        sys.exit(f"INSTAGRAM ERROR: {e.get('message')} (code {e.get('code')}, "
                 f"subcode {e.get('error_subcode')}) {e.get('error_user_msg') or ''}")
    return data


def publish(url, caption):
    ig_id = os.environ["IG_USER_ID"]
    container = ig_call("POST", f"{GRAPH}/{ig_id}/media", image_url=url, caption=caption)["id"]
    for _ in range(20):
        status = ig_call("GET", f"{GRAPH}/{container}", fields="status_code")
        code = status.get("status_code")
        if code == "FINISHED":
            break
        if code in ("ERROR", "EXPIRED"):
            sys.exit(f"Instagram could not process the image: {status}")
        time.sleep(3)
    return ig_call("POST", f"{GRAPH}/{ig_id}/media_publish", creation_id=container)["id"]


def main():
    dry = os.environ.get("DRY_RUN", "false").lower() == "true"
    max_posts = int(os.environ.get("MAX_POSTS", "1"))
    if not dry:
        for name in ("IG_ACCESS_TOKEN", "IG_USER_ID"):
            if not os.environ.get(name):
                sys.exit(f"Missing secret: {name}")
    if not IMAGES_DIR.is_dir():
        sys.exit("No images folder found.")

    todo = next_images(max_posts)
    if not todo:
        log("Nothing left to post: every image in ./images is already in posted.csv.")
        return

    for path in todo:
        log(f"Image: {path.name}" + ("  (DRY RUN)" if dry else ""))
        caption = make_caption(path)
        log("Caption:\n" + caption + "\n")
        if dry:
            log("Dry run: nothing was posted and nothing was recorded.")
            continue
        if caption == FALLBACK_CAPTION and os.environ.get("ALLOW_FALLBACK", "false").lower() != "true":
            sys.exit("No AI caption could be written, so nothing was posted (the image stays in the "
                     "queue and the next scheduled run will try again). Set ALLOW_FALLBACK=true to post "
                     "with the generic caption instead.")
        url = raw_url(path.name)
        if not wait_for_url(url):
            sys.exit(f"The public image link is not available yet: {url}")
        media_id = publish(url, caption)
        record(path.name, media_id, caption)
        log(f"Posted {path.name} (media ID {media_id})")


if __name__ == "__main__":
    main()
