FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        frr \
        frr-pythontools \
        iproute2 \
        iputils-ping \
        mininet \
        net-tools \
        python3 \
        tcpdump \
        traceroute \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /lab

COPY mininet_frr_lab/run_topology.py /lab/run_topology.py
COPY dataset/topologias_convertidas/gabriel/10/0.json /lab/topology.json

CMD ["python3", "/lab/run_topology.py", "--topology", "/lab/topology.json"]
