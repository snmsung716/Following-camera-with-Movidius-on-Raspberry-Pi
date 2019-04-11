#!/bin/bash

# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if [[ $EUID -ne 0 ]]; then
    echo "ERROR: this script must be run as root to install 3rd party packages." >&2
    echo "Please try again with "sudo -E $0", or as root." >&2
    exit 1
fi

params=$@

function yes_or_no {
    if [ "$params" == "-y" ]; then
        return 0
    fi

    while true; do
        read -p "Add third-party Nux Dextop repository and install FFmpeg package (y) / Skip this step (N)" yn
        case $yn in
            [Yy]*) return 0  ;;
            [Nn]*) return  1 ;;
        esac
    done
}

if [[ -f /etc/lsb-release ]]; then
    # Ubuntu
    system_ver=`cat /etc/lsb-release | grep -i "DISTRIB_RELEASE" | cut -d "=" -f2`

    echo
    echo This script installs the following OpenVINO 3rd-party dependencies:
    echo   1. FFmpeg and GStreamer libraries required for OpenCV and Inference Engine
    echo   2. libusb library required for Myriad plugin for Inference Engine
    echo   3. build dependencies for OpenVINO samples
    echo

    apt update
    apt install -y libcairo2-dev libpango1.0-dev libglib2.0-dev libgtk2.0-dev libswscale-dev libavcodec-dev libavformat-dev \
                           libgstreamer1.0-0 gstreamer1.0-plugins-base \
                           build-essential cmake libusb-1.0-0-dev libdrm-dev

    if [ $system_ver = "18.04" ]; then
        sudo -E apt-get install -y libpng-dev
    else
        sudo -E apt-get install -y libpng12-dev
    fi
else
    # CentOS
    echo This script installs the following OpenVINO 3rd-party dependencies:
    echo   1. GStreamer library required for OpenCV and Inference Engine
    echo   2. libusb library required for Myriad plugin for Inference Engine
    echo   3. Python 3.6 for Model Optimizer
    echo   4. gcc 4.8.5 and other build dependencies for OpenVINO samples
    echo
    
    # install libusbx for Inference Engine Myriad plugin and gstreamer for OpenCV
    yum install -y libusbx-devel gstreamer1 gstreamer1-plugins-base
 
    # install gcc to build Inference Engine samples
    yum install -y centos-release-scl epel-release
    yum install -y gcc gcc-c++ make glibc-static glibc-devel libstdc++-static libstdc++-devel libstdc++ libgcc \
                           glibc-static.i686 glibc-devel.i686 libstdc++-static.i686 libstdc++.i686 libgcc.i686 cmake

    # Python 3.6 for Model Optimizer
    yum install -y rh-python36
    source scl_source enable rh-python36

    echo
    echo "Intel(R) Distribution of OpenVINO(TM) toolkit requires FFmpeg for processing audio and video streams with OpenCV. Please select your preferred method for installing FFmpeg:"
    echo
    echo "Option 1: Allow installer script to add a third party repository, Nux Dextop (http://li.nux.ro/repos.html), which contains FFmpeg. FFmpeg rpm package will be installed from this repository. "
    echo "WARNING: This repository is NOT PROVIDED OR SUPPORTED by Intel or CentOS. Neither Intel nor CentOS has control over this repository. Terms governing your use of FFmpeg can be found here: https://www.ffmpeg.org/legal.html "
    echo "Once added, this repository will be enabled on your operating system and can thus receive updates to all packages installed from it. "
    echo
    echo "Consider the following ways to prevent unintended 'updates' from this third party repository from over-writing some core part of CentOS:"
    echo "a) Only enable these archives from time to time, and generally leave them disabled. See: man yum"
    echo "b) Use the exclude= and includepkgs= options on a per sub-archive basis, in the matching .conf file found in /etc/yum.repos.d/ See: man yum.conf"
    echo "c) The yum Priorities plug-in can prevent a 3rd party repository from replacing base packages, or prevent base/updates from replacing a 3rd party package."
    echo
    echo "Option 2: Skip FFmpeg installation."
    echo

    if yes_or_no; then
        rpm -Uvh http://li.nux.ro/download/nux/dextop/el7/x86_64/nux-dextop-release-0-1.el7.nux.noarch.rpm
        yum install -y ffmpeg
    else
        echo "FFmpeg installation skipped. You may build FFmpeg from sources as described here: https://trac.ffmpeg.org/wiki/CompilationGuide/Centos"
        echo
    fi
    exit
fi
