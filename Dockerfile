FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

#Creating dir for app
WORKDIR /app

#Optimization
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

#Dependencies
COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-install-project --no-dev

ENV PATH="/app/.venv/bin:$PATH"

#Copying source code and model
COPY src/ ./src/
COPY models/ ./models/

#Opening port
EXPOSE 8000

#Comand for runing the server
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]