FROM python:3.12-slim

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt  # Create requirements.txt first: pip freeze > requirements.txt

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]