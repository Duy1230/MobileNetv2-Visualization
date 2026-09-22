FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    MNV2_HOST=0.0.0.0 PORT=8501 TORCH_HOME=/home/lab/.cache/torch
WORKDIR /app
COPY requirements.txt .
# CPU wheels keep the image much smaller than CUDA-enabled defaults.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home lab \
    && mkdir -p /home/lab/.cache/torch \
    && chown -R lab:lab /home/lab/.cache
COPY --chown=lab:lab . .
USER lab
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/api/health', timeout=3)" || exit 1
CMD ["python", "interface.py"]
