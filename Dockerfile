FROM python:3.8

#RUN apt-get update && apt-get install -y software-properties-common && \
#    add-apt-repository ppa:ubuntugis/ppa && apt-get update && \
#    apt-get install -y libgdal-dev g++ --no-install-recommends && \
#    apt-get clean -y

## config
ARG USER=dulny
ARG UID=1241

RUN adduser ${USER} --uid ${UID} --home /home/ls6/${USER}/ --disabled-password --gecos "" --no-create-home
RUN mkdir -p /home/ls6/${USER}
RUN chown -R ${USER} /home/ls6/${USER}

USER ${USER}

RUN mkdir -p /home/ls6/${USER}/neuralPDE-2022/

WORKDIR /home/ls6/${USER}/neuralPDE-2022/

COPY ./requirements.txt .
RUN pip install -r requirements.txt

RUN rm requirements.txt

COPY configs/ configs/
COPY src/ src/
COPY scripts/ scripts/

COPY test.py .
COPY train.py .
COPY generate.py .
COPY download.py .