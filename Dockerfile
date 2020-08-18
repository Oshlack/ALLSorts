FROM python:3.8

ENV PATH "/opt:$PATH"

COPY env/requirements.txt /opt/requirements.txt
RUN pip install -r /opt/requirements.txt

# install MoRP
RUN pip install git+https://github.com/breons/MoRP.git
COPY . /opt/
RUN pip install -e /opt/