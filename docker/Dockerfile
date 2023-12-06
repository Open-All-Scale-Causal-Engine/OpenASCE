FROM public.ecr.aws/ubuntu/ubuntu:20.04_stable

RUN mkdir /resources /workspace

WORKDIR /workspace

RUN apt-get update

RUN apt-get install -y build-essential

RUN apt-get install -y wget vim cmake git

RUN wget -O /resources/conda3.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
&& sh /resources/conda3.sh -b -u -f -p /usr/local

RUN conda update -y conda python
