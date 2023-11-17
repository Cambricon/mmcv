#!/bin/bash

# Check if image and command are provided
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <docker_image> <command>"
  exit 1
fi

docker_image="$1"
shift # Remove the first argument (docker_image) from the positional parameters
command_to_run=("$@")

# Create a unique container name
container_name="temp_mmcv_$(date '+%Y%m%d%H%M%S')"

# Run the Docker container
docker run --network=host --name "${container_name}" "${docker_image}" /bin/bash -c "
  set -e
  git clone http://gitlab.software.cambricon.com/neuware/oss/openmmlab/mmcv.git -b $4 /mmcv && 
  cd /mmcv/docs &&
  ${command_to_run[*]}
"

docker cp "${container_name}:/mmcv/docs/" $3

# Clean up: remove the container
docker rm -f "${container_name}" > /dev/null
