FROM python:3.10-bullseye
WORKDIR /app
ADD src ./src
ADD pyproject.toml .
ADD setup.py .

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    rm awscliv2.zip && \
    ./aws/install

RUN pip install --upgrade pip && \
    pip install . --no-cache-dir
