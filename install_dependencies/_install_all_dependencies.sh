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

echo
echo "CV SDK dependencies installation using install_cv_sdk_dependencies.sh"
echo
sudo -E ./install_cv_sdk_dependencies.sh

echo
echo "OpenCL driver installation using install_NEO_OCL_driver.sh"
echo
sudo -E ./install_NEO_OCL_driver.sh

version_gt() {
    # check if first version is greater than second version
    test "$(printf '%s\n' "$@" | sort -V | head -n 1)" != "$1";
}

kernel_version=$(uname -r)
# install kernel 4.14 if current kernel version < 4.13 (GPU NEO driver supports only kernels 4.13.x and higher)
if version_gt "4.13" "$kernel_version" ; then
	echo
	echo "4.14 kernel installation using install_4_14_kernel.sh"
	echo
	sudo -E ./install_4_14_kernel.sh
fi


echo
echo "Next steps:"
echo
if version_gt "$kernel_version" "4.13"; then
	echo "Reboot into installed 4.14 kernel"
	echo
fi
echo "If you use 8th Generation Intel® Core™ processor, you will need to add:"
echo "   i915.alpha_support=1"
echo "   to the 4.14 kernel command line in the grub config file, in order to enable OpenCL functionality for this platform."
echo
echo "Add OpenCL users to the video group: 'sudo usermod -a -G video USERNAME'"
echo "   e.g. if the user running OpenCL host applications is foo, run: sudo usermod -a -G video foo"
echo