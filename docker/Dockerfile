ARG BASE_IMAGE=nvcr.io/nvidia/isaac-sim:6.0.0
ARG LEROBOT_VERSION="<1.0"

FROM ${BASE_IMAGE}

USER root

ARG WORKDIR="/workspaces/openarm_isaac_lab"
ENV WORKDIR=${WORKDIR}
WORKDIR "${WORKDIR}"

RUN apt-get update && apt-get install -y \
  git \
  git-lfs \
  cmake \
  sudo \
  jq \
  ffmpeg \
  python3-pip \
  pipx

###############################
# Install Isaac Lab
###############################
COPY submodules/IsaacLab ${WORKDIR}/submodules/IsaacLab
ENV ISAACLAB_PATH=${WORKDIR}/submodules/IsaacLab
ENV TERM=xterm
RUN ln -s /isaac-sim/ ${WORKDIR}/submodules/IsaacLab/_isaac_sim
RUN for DIR in ${WORKDIR}/submodules/IsaacLab/source/isaaclab*/; do /isaac-sim/python.sh -m pip install --no-deps -e "$DIR"; done
RUN chmod 777 -R /isaac-sim/kit/
RUN chmod a+x /isaac-sim
RUN ${ISAACLAB_PATH}/isaaclab.sh -i
RUN /isaac-sim/python.sh -m pip install -e "${WORKDIR}/submodules/IsaacLab/source/isaaclab_newton[all]"
RUN /isaac-sim/python.sh -m pip install --upgrade pip

###############################
# Install IsaacLab-Arena
###############################
COPY submodules/IsaacLab-Arena ${WORKDIR}/submodules/IsaacLab-Arena
RUN /isaac-sim/python.sh -m pip install -e "${WORKDIR}/submodules/IsaacLab-Arena"

###############################
# Install OpenArm
###############################
COPY source/openarm ${WORKDIR}/source/openarm
RUN /isaac-sim/python.sh -m pip install toml
RUN /isaac-sim/python.sh -m pip install -e "${WORKDIR}/source/openarm"
RUN /isaac-sim/python.sh -m pip install "lerobot${LEROBOT_VERSION}" av

RUN PIPX_HOME=/opt/pipx PIPX_BIN_DIR=/usr/local/bin \
    pipx install "huggingface-hub[cli]" pre-commit

RUN echo "alias python='/isaac-sim/python.sh'" >> /etc/bash.bashrc
RUN echo "alias pip3='/isaac-sim/python.sh -m pip'" >> /etc/bash.bashrc
RUN echo "alias pytest='/isaac-sim/python.sh -m pytest'" >> /etc/bash.bashrc
RUN echo "PS1='[OpenArm Isaac Lab] \[\e[0;32m\]~\u \[\e[0;34m\]\w\[\e[0;36m\] \$ '" >> /etc/bash.bashrc
RUN echo "alias ll='ls -alF --color=auto'" >> /etc/bash.bashrc
RUN echo "alias ..='cd ..'" >> /etc/bash.bashrc
RUN cp /etc/bash.bashrc /root/.bashrc

COPY docker/setup/entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
