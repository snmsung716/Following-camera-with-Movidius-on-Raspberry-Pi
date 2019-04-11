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

echo "Update udev rules so that the toolkit can communicate with your neural compute stick"

if [ -z "$INTEL_CVSDK_DIR" ]; then
    echo "Please setup your environment. Run 'source <OVINO_INSTALLDIR>/bin/setupvars.sh'."
    exit -1
fi

if [ -f "$INTEL_CVSDK_DIR/deployment_tools/inference_engine/external/97-myriad-usbboot.rules" ]; then
    sudo cp "$INTEL_CVSDK_DIR/deployment_tools/inference_engine/external/97-myriad-usbboot.rules" /etc/udev/rules.d/
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    sudo ldconfig
    echo "[install_NCS_udev_rules.sh] udev rules installed"
else
    echo "File '97-myriad-usbboot.rules' is missing. Please check that you installed 'Inference Engine Runtime for Intel® Movidius™ VPU'."
    exit -1
fi 


