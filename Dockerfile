FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY apps/backend ./apps/backend

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir . \
    && addgroup --system --gid 10001 sentinel \
    && adduser --system --uid 10001 --ingroup sentinel sentinel \
    && mkdir -p /runtime_data \
    && chown -R sentinel:sentinel /app /runtime_data

USER sentinel

EXPOSE 8765

CMD ["sentinel", "--bind", "0.0.0.0", "--port", "8765", "--data-dir", "/runtime_data", "--no-open"]
