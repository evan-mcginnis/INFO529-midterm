# Base image for anaconda

FROM continuumio/anaconda3
MAINTAINER "Evan McGinnis"

# Define the anaconda environment
COPY environment.yml .
RUN conda env create -f environment.yml

RUN apt-get update && apt-get install -y libgtk2.0-dev && \
    rm -rf /var/lib/apt/lists/*


# Update and create the conda environment
RUN /opt/conda/bin/conda update -n base -c defaults conda 
RUN /opt/conda/bin/conda create --name midterm

# Activate the midterm environment
RUN echo "conda activate midterm" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# Install the specific versions needed by the midterm
RUN /opt/conda/bin/conda init bash && \
    /opt/conda/bin/conda install python=3.6 && \
    /opt/conda/bin/conda install anaconda-client && \
    /opt/conda/bin/conda install numpy=1.17 tensorflow=1.11 pandas matplotlib -y 

#RUN ["mkdir", "notebooks"]
ENV PYTHONUNBUFFERED=1

# The crop prediction model creation
COPY ./predict-yield.py ./
#COPY ./train.npz ./

# The entrypoint will run the yield prediction.
# Seems a bit indirect, but this is the only way to get things running found to date
COPY ./entrypoint.sh ./

# Jupyter and Tensorboard ports
EXPOSE 8888 6006

ENTRYPOINT ["./entrypoint.sh"]


