FROM mcr.microsoft.com/devcontainers/base:ubuntu-20.04

SHELL [ "bash", "-c" ]

# update apt and install packages
RUN apt update && \
    apt install -yq \
        ffmpeg \
        dkms \
        build-essential

# add user tools
RUN sudo apt install -yq \
        jq \
        jp \
        tree \
        tldr

# add git-lfs and install
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash && \
    sudo apt-get install -yq git-lfs && \
    git lfs install

############################################
# Setup user
############################################

USER vscode

# install azcopy, a tool to copy to/from blob storage
# for more info: https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-blobs-upload#upload-a-file
RUN cd /tmp && \
    wget https://azcopyvnext.azureedge.net/release20230123/azcopy_linux_amd64_10.17.0.tar.gz && \
    tar xvf azcopy_linux_amd64_10.17.0.tar.gz && \
    mkdir -p ~/.local/bin && \
    mv azcopy_linux_amd64_10.17.0/azcopy ~/.local/bin && \
    chmod +x ~/.local/bin/azcopy && \
    rm -rf azcopy_linux_amd64*

# Setup conda
RUN cd /tmp && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash ./Miniconda3-latest-Linux-x86_64.sh -b && \
    rm ./Miniconda3-latest-Linux-x86_64.sh

# Install dotnet
RUN cd /tmp && \
    wget https://dot.net/v1/dotnet-install.sh && \
    chmod +x dotnet-install.sh && \
    ./dotnet-install.sh --channel 7.0 && \
    ./dotnet-install.sh --channel 3.1 && \
    rm ./dotnet-install.sh

