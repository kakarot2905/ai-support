# One container for the whole app: FastAPI serves the API and the built React UI on port 8000.

# 1. Build the React app.
FROM node:22-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# 2. Python backend, which also serves the build from step 1.
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Download Chroma's embedding model (~80 MB) now, so the first upload doesn't wait for it.
RUN python -c "from chromadb.utils.embedding_functions import DefaultEmbeddingFunction as E; E()(['warm up'])"
COPY backend/ backend/
COPY --from=frontend /frontend/dist frontend/dist
WORKDIR /app/backend
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
