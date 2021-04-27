FROM python

COPY . /app

ENV BOTO_CONFIG=/app/source/configs/google.boto

EXPOSE 5000

CMD /app/start-production $PORT
