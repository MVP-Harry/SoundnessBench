FROM ubuntu:24.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    git cmake build-essential python3-dev zlib1g-dev \
    tmux vim wget curl bc libicu-dev psmisc htop libssl-dev \
    && apt-get clean

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p ~/miniconda3 && \
    ~/miniconda3/bin/conda init && \
    rm miniconda.sh

# Ensure shell uses conda
SHELL ["/bin/bash", "-c"]

CMD ["/bin/bash"]
