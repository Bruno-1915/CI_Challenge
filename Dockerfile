FROM tensorflow/tensorflow:2.5.1

COPY requirements.txt /
RUN python3 -m pip install -r /requirements.txt

COPY . /app
WORKDIR /app

EXPOSE $PORT
CMD [ "python" , "app.py"]
