#!/usr/bin/env python3
"""영상 퀴즈 도우미

영상(녹화 파일 / 유튜브 링크)을 넣으면 음성을 글(스크립트)로 바꿔 저장하고,
붙여넣은 퀴즈 문제를 스크립트 내용에 근거해 Claude가 풀어준다.

실행:  python app.py   →  http://127.0.0.1:8765
"""
import atexit
import hashlib
import hmac
import html
import importlib.util
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
import webbrowser
from pathlib import Path
from urllib.parse import urlparse

import requests
from flask import Flask, g, jsonify, make_response, redirect, request, send_from_directory
from werkzeug.exceptions import RequestEntityTooLarge

BASE = Path(__file__).resolve().parent


def load_env_file():
    """같은 폴더의 .env 파일에서 KEY=VALUE 를 읽어 환경변수로 등록한다."""
    p = BASE / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env_file()

DATA = Path(os.environ.get("DATA_DIR") or (BASE / "data"))
VIDEOS = DATA / "videos"
TMP = DATA / "tmp"
for d in (VIDEOS, TMP):
    d.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder=str(BASE / "static"), static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024 * 1024  # 업로드 최대 4GB

MAX_TRANSCRIPT_CHARS = 400_000  # 한 번에 보낼 스크립트 전체 길이 상한
VIDEO_ID_RE = re.compile(r"^[a-f0-9]{10}$")
ADMIN = "admin"  # 영상은 항상 관리자(나)가 올린다. 친구는 내가 공유해 준 영상으로 질문만 한다.
INVITE_CLAIM_HOURS = 24  # 초대 링크를 만들고 나서 이 시간 안에 열어야 함
PUBLIC = {"url": os.environ.get("PUBLIC_URL", "").strip().rstrip("/")}


class UserError(Exception):
    """사용자에게 그대로 보여줄 수 있는 오류."""


# --------------------------------------------------------------------------
# 스크립트 저장소
# --------------------------------------------------------------------------
def save_video(title, source, segments, duration, method, owner=ADMIN):
    vid = uuid.uuid4().hex[:10]
    record = {
        "id": vid,
        "owner": owner,
        "title": title,
        "source": source,
        "method": method,  # "whisper" | "captions"
        "duration": duration,
        "created": int(time.time()),
        "segments": segments,
    }
    (VIDEOS / f"{vid}.json").write_text(
        json.dumps(record, ensure_ascii=False), encoding="utf-8"
    )
    return vid


def load_video(vid, owner=ADMIN):
    """영상 하나를 읽는다. 친구에게 보여줄지는 호출하는 쪽(공유 목록)이 정한다."""
    if not VIDEO_ID_RE.match(vid or ""):
        return None
    p = VIDEOS / f"{vid}.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text(encoding="utf-8"))
    return r if r.get("owner", ADMIN) == owner else None


def list_videos(owner=ADMIN):
    items = []
    for p in VIDEOS.glob("*.json"):
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if r.get("owner", ADMIN) != owner:
            continue
        items.append(
            {
                "id": r["id"],
                "title": r["title"],
                "source": r.get("source", ""),
                "method": r.get("method", ""),
                "duration": r.get("duration", 0),
                "created": r.get("created", 0),
                "chars": sum(len(s["text"]) for s in r["segments"]),
            }
        )
    items.sort(key=lambda x: x["created"], reverse=True)
    return items


def fmt_time(sec):
    sec = int(sec or 0)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def format_transcript(segments, chunk=20):
    """짧은 조각을 약 20초 단위 줄로 합쳐 '[mm:ss] 내용' 형태로 만든다."""
    lines, cur, cur_start = [], [], None
    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        if cur_start is None:
            cur_start = seg["start"]
        cur.append(text)
        if seg.get("end", seg["start"]) - cur_start >= chunk:
            lines.append(f"[{fmt_time(cur_start)}] {' '.join(cur)}")
            cur, cur_start = [], None
    if cur:
        lines.append(f"[{fmt_time(cur_start)}] {' '.join(cur)}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 음성 → 글
# --------------------------------------------------------------------------
def have_module(name):
    return importlib.util.find_spec(name) is not None


def extract_audio(src, dst):
    if not shutil.which("ffmpeg"):
        raise UserError(
            "ffmpeg가 설치되어 있지 않아요. README의 '준비물' 항목을 참고해 설치해 주세요."
        )
    proc = subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", str(dst)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not Path(dst).exists() or Path(dst).stat().st_size < 1000:
        tail = (proc.stderr or "").strip().splitlines()[-1:] or [""]
        raise UserError(
            "영상에서 소리를 꺼내지 못했어요. 영상에 오디오가 녹음되어 있는지 확인해 주세요. "
            f"({tail[0][:120]})"
        )


_whisper_model = None
_whisper_lock = threading.Lock()


def get_whisper():
    global _whisper_model
    with _whisper_lock:
        if _whisper_model is None:
            if not have_module("faster_whisper"):
                raise UserError(
                    "음성 인식 패키지(faster-whisper)가 설치되어 있지 않아요. "
                    "터미널에서 'pip install -r requirements.txt'를 실행해 주세요."
                )
            from faster_whisper import WhisperModel

            size = os.environ.get("WHISPER_MODEL", "small")
            device = os.environ.get("WHISPER_DEVICE", "auto")
            compute = os.environ.get("WHISPER_COMPUTE", "int8")
            _whisper_model = WhisperModel(size, device=device, compute_type=compute)
        return _whisper_model


def transcribe(wav_path, language, job):
    job.update(message="음성 인식 모델 불러오는 중 (처음에는 모델을 내려받느라 오래 걸려요)")
    model = get_whisper()
    job.update(message="음성을 글로 바꾸는 중...", progress=0.1)
    seg_iter, info = model.transcribe(
        str(wav_path),
        language=language or None,
        vad_filter=True,
        beam_size=5,
    )
    duration = float(info.duration or 0)
    segments = []
    for s in seg_iter:
        text = s.text.strip()
        if text:
            segments.append({"start": round(s.start, 1), "end": round(s.end, 1), "text": text})
        if duration:
            job.update(progress=0.1 + 0.88 * min(1.0, s.end / duration))
    return segments, duration


# --------------------------------------------------------------------------
# 유튜브
# --------------------------------------------------------------------------
YT_ID_RE = re.compile(r"(?:v=|youtu\.be/|shorts/|embed/|live/)([A-Za-z0-9_-]{11})")


def parse_youtube_id(url):
    m = YT_ID_RE.search(url or "")
    return m.group(1) if m else None


def fetch_youtube_captions(video_id, langs=("ko", "en")):
    """유튜브 자막(수동/자동생성)을 가져온다. 실패하면 None."""
    if not have_module("youtube_transcript_api"):
        return None
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        segs = []
        if hasattr(YouTubeTranscriptApi, "get_transcript"):  # 구버전 (<1.0)
            for it in YouTubeTranscriptApi.get_transcript(video_id, languages=list(langs)):
                start, dur = float(it["start"]), float(it.get("duration", 0))
                segs.append({"start": round(start, 1), "end": round(start + dur, 1), "text": it["text"]})
        else:  # 신버전 (>=1.0)
            fetched = YouTubeTranscriptApi().fetch(video_id, languages=list(langs))
            for it in fetched:
                start, dur = float(it.start), float(it.duration)
                segs.append({"start": round(start, 1), "end": round(start + dur, 1), "text": it.text})
        segs = [s for s in segs if s["text"].strip()]
        return segs or None
    except Exception:
        return None


def youtube_title(url, video_id):
    try:
        r = requests.get(
            "https://www.youtube.com/oembed",
            params={"url": url, "format": "json"},
            timeout=10,
        )
        if r.ok:
            return r.json().get("title") or f"YouTube {video_id}"
    except Exception:
        pass
    return f"YouTube {video_id}"


def download_youtube_audio(url, outdir, job):
    if not have_module("yt_dlp"):
        raise UserError(
            "이 영상은 자막을 가져오지 못했고, 오디오를 내려받는 패키지(yt-dlp)도 설치되어 있지 않아요. "
            "'pip install -r requirements.txt'를 실행하거나, 화면 녹화 파일로 올려 주세요."
        )
    import yt_dlp

    def hook(d):
        if d.get("status") == "downloading" and d.get("total_bytes"):
            job.update(progress=0.02 + 0.08 * d["downloaded_bytes"] / d["total_bytes"])

    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(outdir / "yt_audio.%(ext)s"),
        "quiet": True,
        "noplaylist": True,
        "progress_hooks": [hook],
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return Path(ydl.prepare_filename(info)), info.get("title")
    except Exception as e:
        raise UserError(f"유튜브 영상을 내려받지 못했어요. 링크를 확인하거나 녹화 파일로 올려 주세요. ({str(e)[:150]})")


# --------------------------------------------------------------------------
# 친구 초대 (1회용 링크 + 이용 시간/횟수 제한)
#
#  - 관리자(나): 이 컴퓨터에서 직접(127.0.0.1) 접속한 사람. 초대장을 만들고 모든 기능을 쓴다.
#  - 게스트(친구): 초대 링크를 '한 번' 열어 시작한 사람. 시작한 순간부터 정해진 시간 동안,
#    정해진 횟수만큼만 질문할 수 있다. 영상은 올릴 수 없고, 내가 초대장에 골라 둔 영상만 쓸 수 있다.
#  - 초대 링크/세션 비밀값은 해시로만 저장한다.
# --------------------------------------------------------------------------
INV_LOCK = threading.RLock()
INV_FILE = DATA / "invites.json"


def _h(s):
    return hashlib.sha256(s.encode()).hexdigest()


def inv_load():
    if not INV_FILE.exists():
        return {}
    try:
        return json.loads(INV_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def inv_save(d):
    tmp = INV_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(INV_FILE)


def invite_status(rec, now=None):
    now = now or time.time()
    if rec.get("revoked"):
        return "revoked"
    if not rec.get("claimed_at"):
        return "waiting" if now < rec["claim_deadline"] else "expired"
    return "active" if now < rec["expires_at"] else "expired"


def invite_public(rec):
    now = time.time()
    st = invite_status(rec, now)
    return {
        "id": rec["id"],
        "label": rec.get("label", ""),
        "status": st,
        "minutes": round(rec["duration"] / 60),
        "remaining_seconds": max(0, int(rec["expires_at"] - now)) if st == "active" else 0,
        "claim_remaining_seconds": max(0, int(rec["claim_deadline"] - now)) if st == "waiting" else 0,
        "max_questions": rec["max_questions"],
        "used_questions": rec["used_questions"],
        "video_count": len(rec.get("video_ids", [])),
    }


def create_invite(label, minutes, max_questions, video_ids):
    token = secrets.token_urlsafe(18)
    iid = uuid.uuid4().hex[:8]
    now = time.time()
    rec = {
        "id": iid, "label": label, "token_hash": _h(token), "created": now,
        "claim_deadline": now + INVITE_CLAIM_HOURS * 3600,
        "duration": minutes * 60, "claimed_at": None, "expires_at": None, "session_hash": None,
        "max_questions": max_questions, "used_questions": 0,
        "video_ids": list(video_ids),  # 이 친구가 질문할 수 있는 영상 (내가 고른 것만)
        "revoked": False, "cleaned": False,
    }
    with INV_LOCK:
        d = inv_load()
        d[iid] = rec
        inv_save(d)
    return token, rec


def find_invite_by_token(token):
    th = _h(token)
    for rec in inv_load().values():
        if hmac.compare_digest(rec["token_hash"], th):
            return rec
    return None


def invite_update(iid, fn):
    """초대장 하나를 잠금 아래에서 고친다. fn(rec) 이 False 를 돌려주면 저장하지 않는다."""
    with INV_LOCK:
        d = inv_load()
        rec = d.get(iid)
        if not rec:
            return None
        if fn(rec) is False:
            return None
        inv_save(d)
        return rec


def revoke_invite(iid):
    with INV_LOCK:
        d = inv_load()
        rec = d.get(iid)
        if not rec:
            return False
        rec["revoked"] = True
        rec["cleaned"] = True
        inv_save(d)
    return True


def cleanup_expired_guests():
    """끝난 초대장을 정리 완료로 표시하고, 오래된 초대장 기록을 지운다."""
    with INV_LOCK:
        d = inv_load()
        changed = False
        for iid, rec in list(d.items()):
            if invite_status(rec) in ("expired", "revoked") and not rec.get("cleaned"):
                rec["cleaned"] = True
                changed = True
            if rec.get("cleaned") and time.time() - rec["created"] > 7 * 86400:
                d.pop(iid)
                changed = True
        if changed:
            inv_save(d)


def janitor_loop():
    while True:
        try:
            cleanup_expired_guests()
        except Exception:
            traceback.print_exc()
        time.sleep(60)


def reserve(owner, field):
    """친구의 질문 횟수를 하나 쓴다. 한도를 넘으면 UserError."""
    if owner == ADMIN:
        return
    mx, used = f"max_{field}", f"used_{field}"
    name = "질문"

    def fn(rec):
        if invite_status(rec) != "active":
            raise UserError("이용 시간이 끝났어요.")
        if rec[used] >= rec[mx]:
            raise UserError(f"{name}을 넣을 수 있는 횟수를 모두 썼어요 ({rec[mx]}회).")
        rec[used] += 1

    if invite_update(owner, fn) is None:
        raise UserError("초대 정보를 찾을 수 없어요.")


def refund(owner, field):
    if owner == ADMIN:
        return
    invite_update(owner, lambda rec: rec.__setitem__(f"used_{field}", max(0, rec[f"used_{field}"] - 1)))


# --- 누가 요청했는지 판별 ---------------------------------------------------
# 외부 터널(cloudflared, ngrok 등)을 통해 들어온 요청은 항상 전달 헤더가 붙고 Host 도 공개 주소다.
# 따라서 "loopback + 로컬 Host + 전달 헤더 없음" 일 때만 관리자로 본다.
_FORWARD_HEADERS = ("X-Forwarded-For", "X-Forwarded-Host", "X-Forwarded-Proto", "Forwarded",
                    "X-Real-Ip", "Cf-Connecting-Ip", "Cf-Ray", "True-Client-Ip", "Via")


# --- 관리자 비밀번호 로그인 (서버에 올렸을 때) ---------------------------------
# ADMIN_PASSWORD 가 설정되어 있으면 /login 에서 비밀번호로 관리자 쿠키를 받는다.
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "").strip()
ADMIN_COOKIE_DAYS = 14
_ADMIN_KEY = hashlib.sha256(("vq-admin-cookie:" + ADMIN_PASSWORD).encode()).digest()
_LOGIN_FAILS = {}  # ip -> [실패 시각...]
_LOGIN_LOCK = threading.Lock()


def _admin_sig(ts):
    return hmac.new(_ADMIN_KEY, str(ts).encode(), hashlib.sha256).hexdigest()


def make_admin_cookie():
    ts = int(time.time())
    return f"{ts}.{_admin_sig(ts)}"


def admin_cookie_ok(c):
    if not ADMIN_PASSWORD or "." not in (c or ""):
        return False
    ts, sig = c.split(".", 1)
    if not ts.isdigit() or time.time() - int(ts) > ADMIN_COOKIE_DAYS * 86400:
        return False
    return hmac.compare_digest(sig, _admin_sig(int(ts)))


def client_ip():
    fwd = request.headers.get("X-Forwarded-For", "")
    return (fwd.split(",")[0].strip() if fwd else request.remote_addr) or "?"


def login_allowed(ip):
    """같은 곳에서 15분에 10번 넘게 틀리면 잠시 막는다."""
    now = time.time()
    with _LOGIN_LOCK:
        fails = [t for t in _LOGIN_FAILS.get(ip, []) if now - t < 900]
        _LOGIN_FAILS[ip] = fails
        return len(fails) < 10


def login_failed(ip):
    with _LOGIN_LOCK:
        _LOGIN_FAILS.setdefault(ip, []).append(time.time())


def is_https():
    return request.headers.get("X-Forwarded-Proto", request.scheme) == "https"


def set_session_cookie(resp, name, value, max_age):
    # https(서버)에서는 SameSite=None 으로 두어 Hugging Face 페이지 안(iframe)에서도 로그인이 유지되게 한다.
    https = is_https()
    resp.set_cookie(name, value, max_age=max_age, httponly=True,
                    samesite="None" if https else "Lax", secure=https)


def request_public_url():
    """서버에 올렸을 때, 지금 접속한 공개 주소(https://xxx.hf.space)를 알아낸다."""
    host = (request.headers.get("X-Forwarded-Host") or request.host or "").split(",")[0].strip()
    if not host or host.split(":")[0] in ("127.0.0.1", "localhost", "0.0.0.0", "[::1]"):
        return ""
    return f"{'https' if is_https() else 'http'}://{host}"


def is_admin_request():
    if admin_cookie_ok(request.cookies.get("vq_admin", "")):
        return True
    if request.remote_addr not in ("127.0.0.1", "::1"):
        return False
    host = request.host.rsplit(":", 1)[0].strip("[]").lower() if not request.host.startswith("[") else "::1"
    if host not in ("127.0.0.1", "localhost", "::1"):
        return False
    return not any(h in request.headers for h in _FORWARD_HEADERS)


def guest_from_cookie():
    """(초대장, None) 또는 (None, 거부 이유 코드)."""
    c = request.cookies.get("vq_guest", "")
    if "." not in c:
        return None, "no_session"
    iid, secret = c.split(".", 1)
    rec = inv_load().get(iid)
    if not rec or not rec.get("session_hash") or not hmac.compare_digest(rec["session_hash"], _h(secret)):
        return None, "no_session"
    st = invite_status(rec)
    if st == "revoked":
        return None, "revoked"
    if st != "active":
        return None, "expired"
    return rec, None


REASON_TEXT = {
    "no_session": "초대 링크로 들어와 주세요. 링크를 받은 친구만 쓸 수 있어요.",
    "expired": "이용 시간이 끝났어요.",
    "revoked": "초대가 취소됐어요.",
}


# --------------------------------------------------------------------------
# 작업(Job) 관리: 영상 처리는 오래 걸리므로 백그라운드로 돌린다
# --------------------------------------------------------------------------
class Job(dict):
    pass


JOBS = {}
SLOT = threading.Semaphore(1)  # 무거운 작업은 한 번에 하나씩


def new_job(owner):
    now = time.time()
    for k in [k for k, v in JOBS.items() if now - v["created"] > 3600]:
        JOBS.pop(k, None)
    j = Job(id=uuid.uuid4().hex[:12], owner=owner, status="queued", message="대기 중", progress=0.0,
            video_id=None, error=None, created=now)
    JOBS[j["id"]] = j
    return j


def start_job(target, *args, owner=ADMIN):
    """영상 처리를 백그라운드로 시작한다."""
    job = new_job(owner)

    def runner():
        workdir = TMP / job["id"]
        try:
            with SLOT:
                job.update(status="running", message="시작하는 중...", progress=0.02)
                workdir.mkdir(parents=True, exist_ok=True)
                target(job, workdir, *args)
            job.update(status="done", progress=1.0, message="완료")
        except UserError as e:
            job.update(status="error", error=str(e), message="실패")
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            job.update(status="error", error=f"예상치 못한 오류가 발생했어요: {e}", message="실패")
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    threading.Thread(target=runner, daemon=True).start()
    return job


def process_upload(job, workdir, src_path, title, language):
    job.update(message="영상에서 소리 꺼내는 중...", progress=0.05)
    wav = workdir / "audio16k.wav"
    extract_audio(src_path, wav)
    segments, duration = transcribe(wav, language, job)
    if not segments:
        raise UserError("말소리를 찾지 못했어요. 소리가 너무 작거나 녹음이 안 된 영상일 수 있어요.")
    job["video_id"] = save_video(title, "upload", segments, duration, "whisper", owner=job["owner"])


def process_youtube(job, workdir, url, language, force_asr):
    vid = parse_youtube_id(url)
    if not vid:
        raise UserError("유튜브 링크 형식이 아니에요. 주소를 다시 확인해 주세요.")
    if not force_asr:
        job.update(message="유튜브 자막 확인하는 중...", progress=0.05)
        langs = (language, "ko", "en") if language else ("ko", "en")
        segs = fetch_youtube_captions(vid, langs)
        if segs:
            duration = segs[-1].get("end", 0)
            job["video_id"] = save_video(youtube_title(url, vid), url, segs, duration, "captions",
                                         owner=job["owner"])
            return
    job.update(message="유튜브 오디오 내려받는 중...", progress=0.02)
    audio, title = download_youtube_audio(url, workdir, job)
    job.update(message="오디오 변환 중...", progress=0.1)
    wav = workdir / "audio16k.wav"
    extract_audio(audio, wav)
    segments, duration = transcribe(wav, language, job)
    if not segments:
        raise UserError("말소리를 찾지 못했어요.")
    job["video_id"] = save_video(title or f"YouTube {vid}", url, segments, duration, "whisper",
                                 owner=job["owner"])


# --------------------------------------------------------------------------
# Claude 로 문제 풀기
# --------------------------------------------------------------------------
SYSTEM_PROMPT = """너는 사용자가 방금 본 영상의 내용을 근거로 퀴즈 문제의 정답을 찾아주는 도우미야.

규칙:
1. <videos> 안의 스크립트는 음성 인식(또는 자동 자막) 결과라서 오타·잘못 들은 단어가 있을 수 있어. 고유명사, 성경 구절, 숫자, 발음이 비슷한 단어는 앞뒤 문맥으로 바로잡아 이해하되, 보정했다면 reasoning에 짧게 밝혀.
2. <quiz>에는 객관식, O/X, 단답형, 빈칸 채우기 등이 섞여 있을 수 있고, 여러 문제가 한꺼번에 붙어 있을 수도 있어. 문제를 하나씩 나눠서 각각 답해.
3. 객관식은 보기 기호(①, 1, A 등 문제에 쓰인 표기 그대로)와 보기 내용을 함께 적어.
4. 정답은 스크립트에서 찾은 내용에 근거해야 해. evidence에는 근거가 되는 부분의 시각([mm:ss])과 스크립트 속 짧은 인용(한 문장 이내)을 넣어.
5. source 표기:
   - video: 스크립트에 정답이 직접 나옴
   - inferred: 직접 나오진 않지만 스크립트 내용에서 합리적으로 추론됨
   - not_in_video: 영상에서 근거를 찾지 못함. 이때는 근거를 지어내지 말고 evidence를 비우고, reasoning에 일반 지식으로 추정했다는 점을 분명히 적어.
6. confidence는 근거가 분명하면 high, 인식 오류나 모호함 때문에 불확실하면 medium/low.
7. 여러 영상이 주어지면 어느 영상에서 찾았는지 evidence의 video에 영상 제목을 적어.
8. 반드시 submit_answers 도구로만 답해."""

ANSWER_TOOL = {
    "name": "submit_answers",
    "description": "문제별 정답과 영상 속 근거를 제출한다.",
    "input_schema": {
        "type": "object",
        "properties": {
            "answers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "description": "문제 번호 또는 구분 (예: '1번')"},
                        "question": {"type": "string", "description": "문제를 한 줄로 요약"},
                        "answer": {"type": "string", "description": "최종 정답. 객관식이면 보기 기호와 내용"},
                        "source": {"type": "string", "enum": ["video", "inferred", "not_in_video"]},
                        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                        "evidence": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "video": {"type": "string"},
                                    "time": {"type": "string", "description": "mm:ss"},
                                    "quote": {"type": "string"},
                                },
                                "required": ["time", "quote"],
                            },
                        },
                        "reasoning": {"type": "string", "description": "왜 그 답인지 1~3문장"},
                    },
                    "required": ["label", "answer", "source", "confidence", "evidence", "reasoning"],
                },
            }
        },
        "required": ["answers"],
    },
}


def ask_claude(videos, quiz):
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise UserError("ANTHROPIC_API_KEY가 설정되지 않았어요. 폴더의 .env 파일에 API 키를 넣어 주세요.")
    model = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")

    parts = []
    for i, v in enumerate(videos, 1):
        title = v["title"].replace('"', "'")
        parts.append(f'<video id="{i}" title="{title}">\n{format_transcript(v["segments"])}\n</video>')
    videos_block = "<videos>\n" + "\n".join(parts) + "\n</videos>"
    if len(videos_block) > MAX_TRANSCRIPT_CHARS:
        raise UserError("선택한 영상의 스크립트가 너무 길어요. 영상을 몇 개만 골라서 다시 시도해 주세요.")

    body = {
        "model": model,
        "max_tokens": 4096,
        "system": SYSTEM_PROMPT,
        "tools": [ANSWER_TOOL],
        "tool_choice": {"type": "tool", "name": "submit_answers"},
        "messages": [
            {
                "role": "user",
                "content": [
                    # 같은 영상으로 여러 번 물어볼 때 비용·시간을 줄이기 위한 캐시 표시
                    {"type": "text", "text": videos_block, "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": f"<quiz>\n{quiz}\n</quiz>"},
                ],
            }
        ],
    }
    try:
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json=body,
            timeout=180,
        )
    except requests.RequestException as e:
        raise UserError(f"Claude 서버에 연결하지 못했어요. 인터넷 연결을 확인해 주세요. ({type(e).__name__})")

    if r.status_code != 200:
        try:
            msg = r.json().get("error", {}).get("message", "")
        except Exception:
            msg = r.text[:200]
        if r.status_code == 401:
            raise UserError("API 키가 올바르지 않아요. .env 파일의 키를 다시 확인해 주세요.")
        if r.status_code == 429:
            raise UserError("요청이 너무 많거나 사용 한도에 걸렸어요. 잠시 후 다시 시도해 주세요.")
        raise UserError(f"Claude 호출에 실패했어요 ({r.status_code}): {msg}")

    for block in r.json().get("content", []):
        if block.get("type") == "tool_use" and block.get("name") == "submit_answers":
            return block["input"].get("answers", []), model
    raise UserError("Claude가 답변 형식을 지키지 않았어요. 한 번 더 시도해 주세요.")


# --------------------------------------------------------------------------
# 라우트
# --------------------------------------------------------------------------
OPEN_PATHS = ("/static/", "/join/", "/login", "/logout")
GUEST_ALLOWED = {("GET", "/api/me"), ("GET", "/api/videos"), ("POST", "/api/answer")}


@app.before_request
def gate():
    # 다른 사이트가 몰래 보내는 요청 차단: Origin 이 있으면 우리 주소와 같아야 한다.
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        origin = request.headers.get("Origin")
        hosts = {request.host, (request.headers.get("X-Forwarded-Host") or "").split(",")[0].strip()}
        if origin and urlparse(origin).netloc not in hosts:
            return jsonify(error="허용되지 않은 요청이에요."), 403

    p = request.path
    if p in ("/", "/favicon.ico") or p.startswith(OPEN_PATHS):
        return None

    g.role, g.owner, g.invite = None, None, None
    if is_admin_request():
        g.role, g.owner = "admin", ADMIN
    else:
        rec, why = guest_from_cookie()
        if not rec:
            return jsonify(error=REASON_TEXT[why], reason=why, admin_login=bool(ADMIN_PASSWORD)), 401
        g.role, g.owner, g.invite = "guest", rec["id"], rec
    # 친구는 '공유된 영상 목록 보기'와 '질문하기'만 가능. 업로드·삭제·스크립트 보기·초대 관리는 모두 막는다.
    if g.role == "guest" and (request.method, p) not in GUEST_ALLOWED:
        return jsonify(error="친구는 쓸 수 없는 기능이에요."), 403
    return None


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/favicon.ico")
def favicon():
    return "", 204


@app.get("/api/me")
def me():
    if g.role == "admin":
        return jsonify(role="admin", public_url=PUBLIC["url"] or request_public_url(),
                       can_logout=admin_cookie_ok(request.cookies.get("vq_admin", "")))
    info = invite_public(g.invite)
    info.pop("label", None)
    info.pop("id", None)
    info["remaining_questions"] = max(0, info["max_questions"] - info["used_questions"])
    return jsonify(role="guest", **info)


@app.get("/api/status")
def status():
    if g.role != "admin":
        return jsonify(error="관리자만 쓸 수 있어요."), 403
    return jsonify(
        api_key=bool(os.environ.get("ANTHROPIC_API_KEY", "").strip()),
        ffmpeg=bool(shutil.which("ffmpeg")),
        whisper=have_module("faster_whisper"),
        yt_dlp=have_module("yt_dlp"),
        captions=have_module("youtube_transcript_api"),
        model=os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5"),
        whisper_model=os.environ.get("WHISPER_MODEL", "small"),
    )


@app.get("/api/videos")
def videos_list():
    if g.role == "guest":  # 친구에게는 내가 공유한 영상의 제목·길이만 보인다
        shared = set(g.invite.get("video_ids", []))
        return jsonify(videos=[
            {k: v[k] for k in ("id", "title", "duration", "chars")}
            for v in list_videos(ADMIN) if v["id"] in shared
        ])
    return jsonify(videos=list_videos(ADMIN))


@app.get("/api/videos/<vid>")
def videos_get(vid):
    r = load_video(vid)
    if not r:
        return jsonify(error="영상을 찾을 수 없어요."), 404
    return jsonify(id=r["id"], title=r["title"], text=format_transcript(r["segments"], chunk=10))


@app.delete("/api/videos/<vid>")
def videos_delete(vid):
    if not VIDEO_ID_RE.match(vid):
        return jsonify(error="잘못된 요청이에요."), 400
    (VIDEOS / f"{vid}.json").unlink(missing_ok=True)
    return jsonify(ok=True)


@app.post("/api/videos/upload")
def videos_upload():
    try:
        f = request.files.get("file")
    except RequestEntityTooLarge:
        return jsonify(error="파일이 너무 커요."), 413
    if not f or not f.filename:
        return jsonify(error="파일을 선택해 주세요."), 400
    language = (request.form.get("language") or "").strip() or None
    title = (request.form.get("title") or "").strip()[:100] or Path(f.filename).stem[:100]

    # 작업 폴더는 start_job 안에서 만들어지므로, 업로드 파일은 별도 임시 위치에 먼저 저장
    stash = TMP / f"upload_{uuid.uuid4().hex[:8]}"
    try:
        stash.mkdir(parents=True, exist_ok=True)
        src = stash / ("input" + Path(f.filename).suffix.lower()[:10])
        f.save(src)
    except Exception:
        shutil.rmtree(stash, ignore_errors=True)
        raise

    def target(job, workdir):
        try:
            process_upload(job, workdir, src, title, language)
        finally:
            shutil.rmtree(stash, ignore_errors=True)

    job = start_job(target)
    return jsonify(job_id=job["id"])


@app.post("/api/videos/youtube")
def videos_youtube():
    data = request.get_json(silent=True) or {}
    url = (data.get("url") or "").strip()
    if not parse_youtube_id(url):
        return jsonify(error="유튜브 링크 형식이 아니에요. 주소를 다시 확인해 주세요."), 400
    language = (data.get("language") or "").strip() or None
    force_asr = bool(data.get("force_asr"))
    job = start_job(process_youtube, url, language, force_asr)
    return jsonify(job_id=job["id"])


@app.get("/api/jobs/<job_id>")
def job_get(job_id):
    j = JOBS.get(job_id)
    if not j:
        return jsonify(error="작업을 찾을 수 없어요."), 404
    return jsonify({k: j[k] for k in ("id", "status", "message", "progress", "video_id", "error")})


@app.post("/api/answer")
def answer():
    data = request.get_json(silent=True) or {}
    quiz = (data.get("question") or "").strip()
    ids = data.get("video_ids") or []
    if not quiz:
        return jsonify(error="문제를 붙여넣어 주세요."), 400
    if not ids:
        return jsonify(error="근거로 사용할 영상을 하나 이상 선택해 주세요."), 400
    if g.role == "guest":
        if len(quiz) > 20000:
            return jsonify(error="문제 글이 너무 길어요. 나눠서 넣어 주세요."), 400
        shared = set(g.invite.get("video_ids", []))
        ids = [i for i in ids if i in shared]  # 내가 공유하지 않은 영상은 쓸 수 없다
    videos = [v for v in (load_video(i) for i in ids) if v]
    if not videos:
        return jsonify(error="선택한 영상을 찾을 수 없어요."), 404
    try:
        if g.role == "guest":
            reserve(g.owner, "questions")
        try:
            answers, model = ask_claude(videos, quiz)
        except BaseException:
            refund(g.owner, "questions")  # 실패한 호출은 횟수에서 빼준다
            raise
    except UserError as e:
        return jsonify(error=str(e)), 400
    return jsonify(answers=answers, model=model)


# --- 관리자 전용: 친구 초대 -------------------------------------------------
def _clamp(v, lo, hi, default):
    try:
        return max(lo, min(hi, int(v)))
    except (TypeError, ValueError):
        return default


@app.get("/api/admin/info")
def admin_info():
    recs = sorted(inv_load().values(), key=lambda r: -r["created"])
    return jsonify(public_url=PUBLIC["url"] or request_public_url(), invites=[invite_public(r) for r in recs])


@app.post("/api/admin/invites")
def admin_invite_create():
    d = request.get_json(silent=True) or {}
    label = (d.get("label") or "").strip()[:40]
    ids = [i for i in dict.fromkeys(x for x in (d.get("video_ids") or []) if isinstance(x, str)) if load_video(i)]
    if not ids:
        return jsonify(error="친구가 문제를 풀 때 쓸 영상을 하나 이상 골라 주세요."), 400
    token, rec = create_invite(
        label,
        _clamp(d.get("minutes"), 5, 1440, 60),
        _clamp(d.get("max_questions"), 1, 200, 10),
        ids,
    )
    return jsonify(token=token, path=f"/join/{token}", invite=invite_public(rec))


@app.delete("/api/admin/invites/<iid>")
def admin_invite_revoke(iid):
    if not revoke_invite(iid):
        return jsonify(error="초대를 찾을 수 없어요."), 404
    return jsonify(ok=True)


# --- 친구가 링크를 여는 곳 -------------------------------------------------
PAGE = """<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="noindex">
<title>영상 퀴즈 도우미</title><style>
body{margin:0;font:16px/1.7 "Apple SD Gothic Neo","Malgun Gothic",system-ui,sans-serif;background:#f5f4f0;color:#1f1d1a;
display:grid;place-items:center;min-height:100vh;padding:20px}
.box{max-width:420px;background:#fff;border:1px solid #e4e1da;border-radius:14px;padding:28px}
h1{font-size:20px;margin:0 0 12px}li{margin:4px 0}.s{color:#716d66;font-size:14px}
button{background:#2f5d50;color:#fff;border:0;border-radius:8px;padding:12px 20px;font:inherit;font-weight:600;cursor:pointer;margin-top:12px}
</style></head><body><div class="box"><h1>__TITLE__</h1>__BODY__</div></body></html>"""


def simple_page(title, body, status=200):
    resp = make_response(PAGE.replace("__TITLE__", title).replace("__BODY__", body), status)
    resp.headers["Cache-Control"] = "no-store"
    return resp


def _join_precheck(token):
    """링크를 지금 시작할 수 있는지 확인. (초대장, None) 또는 (None, 오류 응답)."""
    rec = find_invite_by_token(token)
    if not rec or rec.get("revoked"):
        return None, simple_page("링크를 찾을 수 없어요",
                                 "<p>취소됐거나 잘못된 링크예요. 링크를 보낸 친구에게 다시 받아 주세요.</p>", 404)
    if rec.get("claimed_at"):
        got, _ = guest_from_cookie()
        if got and got["id"] == rec["id"]:
            return rec, redirect("/")  # 같은 브라우저로 다시 열면 그대로 입장
        if invite_status(rec) == "expired":
            return None, simple_page("이용 시간이 끝났어요", "<p>이 링크로 쓸 수 있는 시간이 끝났어요.</p>", 410)
        return None, simple_page("이미 사용된 링크예요",
                                 "<p>이 링크는 한 번만 열 수 있어요. 새 링크가 필요하면 보낸 친구에게 부탁하세요.</p>", 410)
    if time.time() > rec["claim_deadline"]:
        return None, simple_page("링크 기간이 지났어요", "<p>이 링크는 열 수 있는 기간이 지났어요. 새 링크를 받아 주세요.</p>", 410)
    return rec, None


@app.get("/join/<token>")
def join_page(token):
    rec, err = _join_precheck(token)
    if err is not None:
        return err
    # 링크 미리보기(카카오톡·슬랙 등)가 링크를 대신 열어 1회권을 써버리지 않도록,
    # 여기서는 시작하지 않고 '시작하기' 버튼을 눌러야 시작된다.
    body = (
        "<p>공유받은 영상을 바탕으로 퀴즈 정답을 찾아주는 도구예요. 이 링크는 <b>한 번만</b> 열 수 있어요.</p><ul>"
        f"<li>이용 시간: <b>{round(rec['duration'] / 60)}분</b> (시작하기를 누른 순간부터)</li>"
        f"<li>질문: 최대 {rec['max_questions']}회</li>"
        f"<li>공유된 영상: {len(rec.get('video_ids', []))}개 (직접 올릴 수는 없어요)</li></ul>"
        "<p class=\"s\">붙여넣은 문제는 정답을 찾기 위해 AI(Claude)로 전송돼요. "
        "시간이 끝나면 더 이상 쓸 수 없어요. 같은 브라우저로만 계속 쓸 수 있어요.</p>"
        f'<form method="post" action="/join/{html.escape(token)}"><button>시작하기</button></form>'
    )
    return simple_page("영상 퀴즈 도우미에 초대받았어요", body)


@app.post("/join/<token>")
def join_claim(token):
    with INV_LOCK:
        rec, err = _join_precheck(token)
        if err is not None:
            return err
        secret = secrets.token_urlsafe(24)
        now = time.time()
        d = inv_load()
        r = d[rec["id"]]
        r.update(claimed_at=now, expires_at=now + r["duration"], session_hash=_h(secret))
        inv_save(d)
    resp = redirect("/", code=303)
    set_session_cookie(resp, "vq_guest", f"{r['id']}.{secret}", int(r["duration"]) + 3600)
    return resp


# --- 관리자 로그인 ---------------------------------------------------------
LOGIN_FORM = (
    '<form method="post" action="/login">'
    '<input type="password" name="password" autocomplete="current-password" required autofocus '
    'placeholder="관리자 비밀번호" style="width:100%;font:inherit;padding:10px 12px;border:1px solid #e4e1da;border-radius:8px">'
    "<button>로그인</button></form>"
)


@app.get("/login")
def login_page():
    if not ADMIN_PASSWORD:
        return simple_page("로그인을 쓸 수 없어요",
                           "<p>서버에 <b>ADMIN_PASSWORD</b>가 설정되어 있지 않아요. "
                           "Hugging Face Space의 Settings → Secrets에 추가한 뒤 다시 시작해 주세요.</p>", 503)
    return simple_page("관리자 로그인", "<p class=\"s\">영상을 올리고 친구를 초대하는 사람만 로그인해요.</p>" + LOGIN_FORM)


@app.post("/login")
def login_submit():
    if not ADMIN_PASSWORD:
        return redirect("/login", code=303)
    ip = client_ip()
    if not login_allowed(ip):
        return simple_page("잠시 후 다시 시도해 주세요", "<p>비밀번호를 여러 번 틀렸어요. 15분 뒤에 다시 시도해 주세요.</p>", 429)
    pw = request.form.get("password", "")
    if not hmac.compare_digest(pw.encode(), ADMIN_PASSWORD.encode()):
        login_failed(ip)
        return simple_page("관리자 로그인", '<p style="color:#a3321f">비밀번호가 맞지 않아요.</p>' + LOGIN_FORM, 401)
    resp = redirect("/", code=303)
    set_session_cookie(resp, "vq_admin", make_admin_cookie(), ADMIN_COOKIE_DAYS * 86400)
    return resp


@app.post("/logout")
def logout():
    resp = redirect("/login" if ADMIN_PASSWORD else "/", code=303)
    resp.delete_cookie("vq_admin")
    return resp


# --------------------------------------------------------------------------
# 외부 공유(터널)
# --------------------------------------------------------------------------
def start_tunnel(port):
    """cloudflared 임시 터널을 켜서 친구가 접속할 공개 주소를 만든다."""
    exe = shutil.which("cloudflared")
    if not exe:
        print("[공유] cloudflared 가 설치되어 있지 않아요. README의 '친구에게 공유하기'를 참고해 설치해 주세요.\n")
        return
    proc = subprocess.Popen(
        [exe, "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
    )
    atexit.register(lambda: proc.poll() is None and proc.terminate())

    def reader():
        for line in proc.stdout:
            m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
            if m and PUBLIC["url"] != m.group(0):
                PUBLIC["url"] = m.group(0)
                print(f"\n[공유] 친구용 공개 주소가 열렸어요: {PUBLIC['url']}")
                print("[공유] 화면 맨 아래 '친구 초대'에서 1회용 링크를 만들어 보내세요. 이 창을 닫으면 공유도 끝나요.\n")

    threading.Thread(target=reader, daemon=True).start()


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8765"))
    url = f"http://{host if host != '0.0.0.0' else '127.0.0.1'}:{port}"
    threading.Thread(target=janitor_loop, daemon=True).start()
    if "--share" in sys.argv or os.environ.get("SHARE") == "1":
        start_tunnel(port)
    print(f"\n영상 퀴즈 도우미가 켜졌어요 → {url}\n(끄려면 이 창에서 Ctrl+C)\n")
    if os.environ.get("NO_BROWSER") != "1":
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    if have_module("waitress"):
        from waitress import serve

        serve(app, host=host, port=port, threads=8,
              max_request_body_size=app.config["MAX_CONTENT_LENGTH"], channel_timeout=600)
    else:
        app.run(host=host, port=port, threaded=True)
