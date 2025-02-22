# Etapa 1: Build com dependências necessárias
FROM python:3.9-slim AS builder

# Define o diretório de trabalho
WORKDIR /app

# Atualiza e instala dependências do sistema necessárias
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev && \
    rm -rf /var/lib/apt/lists/*

# Instala dependências do Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Etapa 2: Imagem final leve
FROM python:3.9-slim

# Define o diretório de trabalho
WORKDIR /app

# Copia apenas os arquivos essenciais da etapa de build
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copia o código do projeto
COPY . .

# Expõe a porta do Flask
EXPOSE 5000

# Comando de execução do programa
CMD ["python", "app.py"]
