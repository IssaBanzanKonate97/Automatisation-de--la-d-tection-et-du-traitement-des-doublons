FROM python:3
ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
WORKDIR /usr/src/app
COPY requirements.txt ./
COPY anomalieV2.json ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD [ "python", "./Automatisation.py" ]