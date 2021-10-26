FROM continuumio/anaconda3
MAINTAINER "Evan McGinnis"

COPY environment.yml .
RUN conda env create -f environment.yml

RUN apt-get update && apt-get install -y libgtk2.0-dev && \
    rm -rf /var/lib/apt/lists/*

RUN /opt/conda/bin/conda update -n base -c defaults conda 
#RUN conda config --set allow_conda_downgrades true
#RUN conda install conda=4.6.14
RUN /opt/conda/bin/conda create --name midterm

RUN echo "conda activate midterm" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

RUN /opt/conda/bin/conda init bash && \
    /opt/conda/bin/conda install python=3.6 && \
    /opt/conda/bin/conda install anaconda-client && \
    /opt/conda/bin/conda install numpy=1.17 tensorflow=1.9 pandas matplotlib -y 

RUN ["mkdir", "notebooks"]
COPY ./predict-yield.py ./
COPY ./train.npz ./
COPY ./entrypoint.sh ./

# Jupyter and Tensorboard ports
EXPOSE 8888 6006

ENTRYPOINT ["./entrypoint.sh"]
#ENTRYPOINT ["/bin/bash"]


