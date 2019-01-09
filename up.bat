@echo off
docker run  -v "%cd%":/app --rm -it jfloff/alpine-python:3.6 -- bash -c "apk add jpeg-dev zlib-dev; cd app; pip3 install -r requirements.txt; exec bash"