FROM tiangolo/meinheld-gunicorn:python3.6 as builder

WORKDIR /app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# lint
RUN pip install --upgrade pip
# RUN pip install flake8
# COPY . /app
# RUN flake8 --ignore=E501,F401 .

# install python dependencies
COPY ./requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt


FROM tiangolo/meinheld-gunicorn:python3.6

COPY ./app /app

WORKDIR /app

# install dependencies
COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache /wheels/*
RUN [ "python", "-c", "import nltk; nltk.download('stopwords')" ]
