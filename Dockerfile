FROM python:3.8-slim

COPY . .

RUN pip3 install flask gunicorn numpy sklearn pandas
