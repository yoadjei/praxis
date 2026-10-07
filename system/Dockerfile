# PRAXIS api and worker image.
#
# 3.14, matching the interpreter the numbers are actually produced on, and NOT the 3.11 that §2
# pins. This used to say that the container was the one place the pin cost nothing to honour.
# That was wrong, and it was wrong in a way nobody could see until the image was first built:
# requirements.lock was compiled on 3.14 and pins contourpy==1.4.0, which requires >=3.12, so
# `pip install --require-hashes -r requirements.lock` on python:3.11 fails to resolve and the
# image cannot be built at all. Honouring the pin cost the whole build.
#
# Changing the base to 3.14 rather than recompiling the lock for 3.11 is deliberate. D11 already
# defers the pin and says no local number may be called reproducible on the pinned interpreter;
# until a 3.11 environment exists, a container on a different interpreter from the one the
# candidate runs would reproduce nothing. One interpreter across local and container is what R7
# asks for. Recompiling the lock against 3.11 moves every pin and is the other half of closing
# D11 - it belongs with that decision, not with a Dockerfile edit.
#
# R6: nothing is fetched at runtime. ULTRALYTICS_OFFLINE and HF_HUB_OFFLINE stop the two libraries
# that would otherwise reach out on first use, and the weights are MOUNTED rather than baked - see
# the note above the runtime stage.

# --- frontend -------------------------------------------------------------------------------
# Built in its own stage so node and 200 MB of node_modules never reach the runtime image. The
# bundle is static once built; nothing in the runtime stage needs npm.
FROM node:22-alpine AS frontend

WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
# `npm ci` and not `npm install`: it installs exactly the lock file, which is what makes two
# builds of one commit produce the same bundle. R7 applies to the browser half too.
RUN npm ci
COPY frontend/ ./
RUN npm run build


# --- runtime --------------------------------------------------------------------------------
FROM python:3.14-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ULTRALYTICS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    TORCH_HOME=/app/weights/torch \
    PRAXIS_SERVE_FRONTEND=1

# ffmpeg for Phase 1 metadata and Phase 3 decoding; libgl and libglib for OpenCV.
RUN apt-get update && apt-get install --no-install-recommends -y \
        ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first so a source edit does not reinstall torch.
COPY requirements.linux-cpu.lock requirements.txt ./
# The linux-cpu lock, not requirements.lock. The latter was compiled `--universal`, so on linux
# it resolves torch to the CUDA build and brings seventeen nvidia-* packages with it - gigabytes
# of libraries into a container with no GPU, which D10 says is CPU-only anyway. The header of
# requirements.linux-cpu.lock explains the whole of it, including why the pinned versions have
# not moved. The extra index is where the CPU wheels live; hashes are still required, so a wheel
# that is not the one this lock names cannot be installed from it.
#
# The fallback stays for the case where the lock has not been generated: unhashed, loud about it,
# and better than an image that cannot be built at all.
RUN if [ -s requirements.linux-cpu.lock ] && grep -qv '^#' requirements.linux-cpu.lock; then \
        pip install --require-hashes \
            --extra-index-url https://download.pytorch.org/whl/cpu \
            -r requirements.linux-cpu.lock; \
    else \
        echo "WARNING: requirements.linux-cpu.lock is unpopulated; installing unhashed pins" >&2; \
        pip install --extra-index-url https://download.pytorch.org/whl/cpu \
            -r requirements.txt; \
    fi

COPY praxis/ ./praxis/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY migrations/ ./migrations/
COPY alembic.ini ./
COPY --from=frontend /build/dist ./frontend/dist

# Weights are NOT copied. `weights/` in the repository holds a .gitkeep and nothing else, so a
# COPY of it produced an image that imported cleanly and then refused every pose call - R6 forbids
# fetching them at runtime, so the failure had no recovery. They are 200 MB of binary that belongs
# in neither a git history nor a layer, and a reader who wants to verify a checksum needs the file
# on disk anyway. docker-compose mounts B:/praxis/weights here read-only. D58 makes vendoring an
# operator step; this keeps it one.
RUN mkdir -p /app/weights /app/weights/torch

# Nothing runs as root that does not have to.
RUN useradd --create-home --uid 10001 praxis && chown -R praxis:praxis /app
USER praxis

EXPOSE 8000
CMD ["uvicorn", "praxis.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
