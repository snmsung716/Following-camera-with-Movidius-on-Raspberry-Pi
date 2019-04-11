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

INSTALLDIR=/home/pi/Downloads/inference_engine_vpu_arm
export INTEL_CVSDK_DIR=$INSTALLDIR

# parse command line options
while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    -pyver)
    python_version=$2
    echo python_version = "${python_version}"
    shift
    ;;
    *)
    # unknown option
    ;;
esac
shift
done

if [ -e $INSTALLDIR/openvx ]; then
    export LD_LIBRARY_PATH=$INSTALLDIR/openvx/lib:$LD_LIBRARY_PATH
fi

if [ -e $INSTALLDIR/deployment_tools/model_optimizer ]; then
    export LD_LIBRARY_PATH=$INSTALLDIR/deployment_tools/model_optimizer/bin:$LD_LIBRARY_PATH
fi

if [ -e $INSTALLDIR/deployment_tools/inference_engine ]; then
    export InferenceEngine_DIR=$INTEL_CVSDK_DIR/deployment_tools/inference_engine/share

    if [[ -f /etc/centos-release ]]; then
        IE_PLUGINS_PATH=$INTEL_CVSDK_DIR/deployment_tools/inference_engine/lib/centos_7.4/intel64
    elif [[ -f /etc/lsb-release ]]; then
         UBUNTU_VERSION=$(lsb_release -r -s)
         if [[ $UBUNTU_VERSION = "16.04" ]]; then
                IE_PLUGINS_PATH=$INTEL_CVSDK_DIR/deployment_tools/inference_engine/lib/ubuntu_16.04/intel64
         elif [[ $UBUNTU_VERSION = "18.04" ]]; then
                IE_PLUGINS_PATH=$INTEL_CVSDK_DIR/deployment_tools/inference_engine/lib/ubuntu_18.04/intel64
         elif cat /etc/lsb-release | grep -q "Yocto" ; then
                IE_PLUGINS_PATH=$INTEL_CVSDK_DIR/deployment_tools/inference_engine/lib/ubuntu_16.04/intel64
         fi
    elif [[ -f /etc/os-release ]]; then
         OS_NAME=$(lsb_release -i -s)
         OS_VERSION=$(lsb_release -r -s)
         OS_VERSION=${OS_VERSION%%.*}
         if [ $OS_NAME = "Raspbian" ] && [ $OS_VERSION = "9" ]; then
                IE_PLUGINS_PATH=$INTEL_CVSDK_DIR/deployment_tools/inference_engine/lib/raspbian_9/armv7l
         fi
    fi

    if [[ -e ${IE_PLUGINS_PATH}/arch_descriptions ]]; then
        export ARCH_ROOT_DIR=${IE_PLUGINS_PATH}/arch_descriptions
    fi

    export HDDL_INSTALL_DIR=$INSTALLDIR/deployment_tools/inference_engine/external/hddl
    export LD_LIBRARY_PATH=/opt/intel/opencl:$HDDL_INSTALL_DIR/lib:$INSTALLDIR/deployment_tools/inference_engine/external/gna/lib:$INSTALLDIR/deployment_tools/inference_engine/external/mkltiny_lnx/lib:$INSTALLDIR/deployment_tools/inference_engine/external/omp/lib:$IE_PLUGINS_PATH:$LD_LIBRARY_PATH
fi

if [ -e "$INSTALLDIR/opencv" ]; then
if [ -f "$INSTALLDIR/opencv/setupvars.sh" ]; then
    source "$INSTALLDIR/opencv/setupvars.sh"
else
    export OpenCV_DIR="$INSTALLDIR/opencv/share/OpenCV"
    export LD_LIBRARY_PATH="$INSTALLDIR/opencv/lib:$LD_LIBRARY_PATH"
    export LD_LIBRARY_PATH="$INSTALLDIR/opencv/share/OpenCV/3rdparty/lib:$LD_LIBRARY_PATH"
fi
fi

export PATH="$INTEL_CVSDK_DIR/deployment_tools/model_optimizer:$PATH"
export PYTHONPATH="$INTEL_CVSDK_DIR/deployment_tools/model_optimizer:$PYTHONPATH"

if [ -z "$python_version" ]; then
    if command -v python3.6 >/dev/null 2>&1; then
        python_version=3.6
        python_bitness=$(python3.6 -c 'import sys; print(64 if sys.maxsize > 2**32 else 32)')
    elif command -v python3.5 >/dev/null 2>&1; then
        python_version=3.5
        python_bitness=$(python3.5 -c 'import sys; print(64 if sys.maxsize > 2**32 else 32)')
    elif command -v python3.4 >/dev/null 2>&1; then
        python_version=3.4
        python_bitness=$(python3.4 -c 'import sys; print(64 if sys.maxsize > 2**32 else 32)')
    elif command -v python2.7 >/dev/null 2>&1; then      
        python_version=2.7
    elif command -v python >/dev/null 2>&1; then
        python_version=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    fi
fi

if [ "$python_bitness" != "" ] && [ "$python_bitness" != "64" ] && [ "$OS_NAME" != "Raspbian" ]; then
    echo "[setupvars.sh] 64 bitness for Python" $python_version "is requred"
fi

if [ ! -z "$python_version" ]; then
    if [[ -f /etc/centos-release ]]; then
        if [[ -e $INTEL_CVSDK_DIR/python/python$python_version/centos7 ]]; then
            export PYTHONPATH="$INTEL_CVSDK_DIR/python/python$python_version:$INTEL_CVSDK_DIR/python/python$python_version/centos7:$PYTHONPATH"
        fi
    elif [[ -f /etc/lsb-release ]]; then
        if [[ -e $INTEL_CVSDK_DIR/python/python$python_version/ubuntu16 ]]; then
            export PYTHONPATH="$INTEL_CVSDK_DIR/python/python$python_version:$INTEL_CVSDK_DIR/python/python$python_version/ubuntu16:$PYTHONPATH"
        elif [[ -e $INTEL_CVSDK_DIR/python/python$python_version/ubuntu18 ]]; then
            export PYTHONPATH="$INTEL_CVSDK_DIR/python/python$python_version:$INTEL_CVSDK_DIR/python/python$python_version/ubuntu18:$PYTHONPATH"
        fi
    elif [[ -f /etc/os-release ]] && [ $(lsb_release -i -s) = 'Raspbian' ]; then
        if [[ -e $INTEL_CVSDK_DIR/python/python$python_version/armv7l ]]; then
            export PYTHONPATH="$INTEL_CVSDK_DIR/python/python$python_version:$INTEL_CVSDK_DIR/python/python$python_version/armv7l:$PYTHONPATH"
        fi
    fi
fi

echo "[setupvars.sh] OpenVINO environment initialized"
