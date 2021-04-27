FROM ubuntu:20.04

COPY . /app

RUN apt-get update && apt-get install -y python3-pip && pip3 install -r /app/requirements.txt

CMD /app/start-production

