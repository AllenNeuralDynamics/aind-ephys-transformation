FROM python:3.10-bullseye
WORKDIR /app
ADD src ./src
ADD pyproject.toml .
ADD setup.py .

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    rm awscliv2.zip && \
    ./aws/install

# Add git in case we need to install from branches
RUN apt-get update && apt-get install -y git && \
    pip install --upgrade pip && \
    pip install . --no-cache-dir
