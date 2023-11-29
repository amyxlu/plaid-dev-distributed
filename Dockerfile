FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-devel
RUN conda install -c anaconda git wget --yes
RUN git clone https://github.com/amyxlu/openfold.git ~/openfold
RUN python ~/openfold/setup.py develop