# Two stages. The first installs the packages and turns the raw file into the clean
# table. The second is the image that runs: it gets only the packages, the clean
# table and the code, so the 74 MB raw file and the build steps are not shipped.
#
#   docker build -t product-similarity .
#   docker run --rm -p 8000:8000 product-similarity
#   then open http://localhost:8000/docs

# ---- stage 1: build ----------------------------------------------------------
FROM python:3.10-slim AS build
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Packages go into a virtual environment so stage 2 can copy them in one piece.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Requirements before code: Docker reuses this step until requirements.txt changes,
# so editing the code does not reinstall every package.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Stage 01, run inside the image, so the table always matches the code.
COPY similarity/ similarity/
COPY data/archive/ data/archive/
RUN python -m similarity.preprocessing

# ---- stage 2: run ------------------------------------------------------------
FROM python:3.10-slim
WORKDIR /app
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# An ordinary user instead of root. A number, not a name, so Kubernetes can check
# that it is not root (runAsNonRoot).
RUN useradd --uid 10001 --no-create-home app

# Least often changed first, so a code edit rebuilds only the last step.
COPY --from=build /opt/venv /opt/venv
COPY --from=build /app/data/processed/ data/processed/
COPY similarity/ similarity/
COPY app.py .

USER 10001
EXPOSE 8000

# Docker's own check, shown as "healthy" in `docker ps`. Kubernetes uses its own probes.
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"]

# One process per container. More traffic means more containers, not more processes.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
