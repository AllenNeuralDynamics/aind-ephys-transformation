FROM python:3.10-bullseye
WORKDIR /app
ADD src ./src
ADD pyproject.toml .
ADD setup.py .

# Add git in case we need to install from branches
RUN apt-get update && apt-get install -y git
RUN pip install --upgrade pip
RUN pip install .[ephys] --no-cache-dir
