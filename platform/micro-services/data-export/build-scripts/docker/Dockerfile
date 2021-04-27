FROM python
RUN pip install redis
COPY . /app

CMD python /app/source/google/export_worker.py
