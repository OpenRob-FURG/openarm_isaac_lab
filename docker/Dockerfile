ARG BASE_IMAGE=nvcr.io/nvidia/isaac-sim:6.0.0-dev2
# NOTE ON THE BASE IMAGE PIN
#
# This must stay aligned with the pin in
# submodules/IsaacLab-Arena/docker/Dockerfile.isaaclab_arena. IsaacLab-Arena
# and the IsaacLab submodule it vendors are tested against the -dev2 build.
#
# The 6.0.0 *release* build (6.0.0-rc.59+release.41464) is NOT compatible with
# them. On that base the physics tensor view is invalidated by the reset that
# PhysxManager performs and the cached view is never rebuilt, so every
# articulation fails to initialise:
#
#   [Warning] [omni.physx.tensors.plugin] All physics information was deleted
#             while being used by a tensor view class. ... was invalidated.
#   [Error]   [omni.physx.tensors.plugin] Simulation view object is invalidated
#             and cannot be used again to call createArticulationView
#   RuntimeError: Failed to create articulation at: /World/envs/env_.*/Robot/
#                 root_joint. Please check PhysX logs.
#
# The visible symptom at the environment level is:
#   Exception: Failed to get DOF velocities from backend
#
# That reproduces for Arena's own stock environment (lift_object) and for a
# minimal pure-IsaacLab InteractiveScene probe, so it is not a task-definition
# problem and no source change in this repository can work around it.
# 6.0.0 also renamed the physics tensors extension from `omni.physics.tensors`
# to `omni.physx.tensors` and dropped its `impl` python submodule, which
# IsaacLab-Arena's physx interop layer still expects.
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

# IsaacLab-Arena's registries eagerly import the G1 embodiment, and
# isaaclab_arena.remote_policy needs a ZMQ stack, so both are required even by
# projects that never touch a humanoid or a remote policy.
RUN /isaac-sim/python.sh -m pip install pyzmq msgpack==1.1.0 msgpack-numpy

# Pinocchio ships as a cmeel wheel inside the Pink extension's pip_prebundle,
# but that cmeel prefix is not on sys.path, so `import pinocchio` fails even
# though libpinocchio is installed. Put the prefix on the path via a .pth file
# in the site-packages that Isaac Sim actually uses.
RUN PINK_PREFIX="/isaac-sim/exts/isaacsim.robot_motion.pink/pip_prebundle/cmeel.prefix/lib/python3.12/site-packages"; \
    SITE="$(/isaac-sim/python.sh -c 'import isaaclab, os, sysconfig; print(sysconfig.get_paths()["purelib"])')"; \
    if [ -d "$PINK_PREFIX" ]; then echo "$PINK_PREFIX" > "$SITE/cmeel_pinocchio.pth"; \
    else echo "WARNING: cmeel pinocchio prefix not found at $PINK_PREFIX" >&2; fi

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
