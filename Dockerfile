# Dockerfile para despliegue en Render
FROM python:3.10-slim

# Configurar variables de entorno para Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app:$PYTHONPATH" \
    PORT=8080

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Copiar requirements y instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación
COPY . .

# Crear directorios necesarios
RUN mkdir -p artifacts logs static templates

# Descargar datos de NLTK necesarios
RUN python -c "import nltk; nltk.download('punkt')"

# Exponer el puerto (Render usa la variable PORT)
EXPOSE $PORT

# Comando para ejecutar la aplicación
# Render requiere que la app escuche en 0.0.0.0 y use la variable PORT
CMD ["python", "-c", "import uvicorn; from app import app; uvicorn.run(app, host='0.0.0.0', port=int(__import__('os').environ.get('PORT', 8080)))"]