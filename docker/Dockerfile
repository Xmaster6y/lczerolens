# syntax=docker/dockerfile:1
FROM python:3.9.18

WORKDIR /code

RUN apt-get update && apt-get install -y \
    ocl-icd-opencl-dev \
    libopenblas-dev \
    zip

COPY poetry.lock pyproject.toml /code/
RUN pip install --upgrade pip
RUN pip install poetry==1.6.1
RUN poetry config virtualenvs.create false
RUN poetry install --no-interaction --no-ansi --with demo

RUN mkdir -p /service/demo
COPY demo /service/demo
EXPOSE 8000
COPY docker/start.sh ./
RUN chmod +x start.sh
ENTRYPOINT ["/code/start.sh"]
