FROM tensorflow/tensorflow:2.5.1

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt /
RUN python3 -m pip install -r /requirements.txt

COPY . /app
WORKDIR /app

CMD [ "python" , "app.py"]
