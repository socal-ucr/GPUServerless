FROM nvidia/cuda:10.1-devel

ARG ADDITIONAL_PACKAGE
# Alternatively use ADD https:// (which will not be cached by Docker builder)
RUN apt-get -qq update && apt-get -qq -y install apt-utils ${ADDITIONAL_PACKAGE}
RUN apt-get -qq -y install curl python3 python3-pip \
    && echo "Pulling watchdog binary from Github." \
    && curl -sSL https://github.com/openfaas/faas/releases/download/0.13.0/fwatchdog > /usr/bin/fwatchdog \
    && chmod +x /usr/bin/fwatchdog \
    && apt -qq -y purge curl

ENV PATH=$PATH:/home/app/.local/bin

WORKDIR /home/app/

COPY index.py           .
COPY requirements.txt   .
RUN pip3 install --user -r requirements.txt

RUN mkdir -p function
RUN touch ./function/__init__.py
WORKDIR /home/app/function/
#COPY function/requirements.txt  .
#RUN pip install --user -r requirements.txt
WORKDIR /home/app/
COPY function           function

ENV fprocess="python3 index.py"
EXPOSE 8080

HEALTHCHECK --interval=3s CMD [ -e /tmp/.lock ] || exit 1

CMD ["fwatchdog"]

