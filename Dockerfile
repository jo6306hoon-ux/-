# Hugging Face Spaces (Docker) 용 이미지
FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# Spaces 는 UID 1000 사용자로 실행한다
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    HOST=0.0.0.0 \
    PORT=7860 \
    NO_BROWSER=1 \
    WHISPER_MODEL=small \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# 음성 인식 모델을 미리 받아 둔다 (첫 분석이 오래 걸리지 않도록)
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', compute_type='int8')"

COPY --chown=user . .

EXPOSE 7860
CMD ["python", "app.py"]
