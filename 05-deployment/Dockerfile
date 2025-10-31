FROM agrigorev/zoomcamp-model:2025
# add your stuff here

RUN pip install uv
# COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /code

# ENV PATH="/code/.venv/bin:$PATH"

COPY "pyproject.toml" "uv.lock" "./"
RUN uv sync --locked

COPY "predict.py" "pipeline_v1.bin" "./"

EXPOSE 9696

ENTRYPOINT ["uv", "run", "uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]