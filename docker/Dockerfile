# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/go/dockerfile-reference/

# Want to help us make this template better? Share your feedback here: https://forms.gle/ybq9Krt8jtBL3iCk7

ARG PYTHON_VERSION=3.11.5
FROM python:${PYTHON_VERSION}-slim as builder

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install cv2 dependencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Setup uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN uv venv --python ${PYTHON_VERSION}

# Install requirements without project
COPY pyproject.toml uv.lock /app/
RUN uv sync --frozen --no-install-project --no-dev

# Copy required modules then install
COPY src/       /app/src/
COPY configs/   /app/configs/
COPY models/    /app/models/
COPY README.md  /app/
RUN uv sync --frozen --no-dev

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/go/dockerfile-user-best-practices/
# ARG UID=10001
# RUN adduser \
#     --disabled-password \
#     --gecos "" \
#     --home "/nonexistent" \
#     --shell "/sbin/nologin" \
#     --no-create-home \
#     --uid "${UID}" \
#     appuser

# # Switch to the non-privileged user to run the application.
# USER appuser

# Copy files from builder
# COPY --from=builder --chown=appuser:appuser /app /app

# Make sure the executables in .venv are accessible
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"


# Copy the source code into the container.
# Expose the port that the application listens on.
EXPOSE 8000

# Run the application.
CMD python src/gradio_ui/main.py
