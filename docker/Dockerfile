FROM nvcr.io/nvidia/pytorch:23.01-py3
RUN pip install tensorboardX
RUN git clone https://github.com/Verified-Intelligence/auto_LiRPA.git; cd auto_LiRPA; git checkout 58ee1c18764d8860bb8f690e809cd9128d81eb93; python setup.py install; python auto_LiRPA/cuda_utils.py install; cd ..
RUN pip install git+https://github.com/fra31/auto-attack
RUN pip install webdataset==0.2.31


