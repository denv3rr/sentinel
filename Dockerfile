FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY apps/backend ./apps/backend

RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir .

EXPOSE 8765

CMD ["sentinel", "--bind", "0.0.0.0", "--port", "8765", "--no-open"]