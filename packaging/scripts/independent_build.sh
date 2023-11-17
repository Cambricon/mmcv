#!/bin/bash
set -e

ABS_DIR_PATH=$(dirname $(readlink -f $0))
IMAGE_WHEEL_NAME="yellow.hub.cambricon.com/mmcv/base/x86_64/mmcv"
DOCKER_FILE="$ABS_DIR_PATH/../mlu_docker/Dockerfile"
RELEASE_TYPE=NULL

while getopts "r:p:m:v:w:t:f:" opt
do
  case $opt in
    r)
	      RELEASE_TYPE=$OPTARG;;
    p)
	      PYTHON_VERSION=$OPTARG;;
    m)
	      MMCV_BRANCH=$OPTARG;;
    v)
              PYTORCH_VERSION=$OPTARG;;
    w)
	      IMAGE_WHEEL_NAME=$OPTARG;;
    t)
	      TAG=$OPTARG;;
    f)
	      DOCKER_FILE=$OPTARG;;
    ?)
	      echo "there is unrecognized parameter."
	      exit 1;;
  esac
done

build_all_wheels_func() {
  sleep 1

  # Read catch version from build.property file
  CATCH_VERSION=$(grep -o '"catch_version": \["[^"]*' "$ABS_DIR_PATH/build.property" | sed 's/"catch_version": \["//')

  # Validate required variables
  if [[ -z "$MMCV_BRANCH" ]]; then
    echo "MMCV branch must be specified using -m"
    exit 1
  fi

  # Set default values if not provided
  PYTHON_VERSION="${PYTHON_VERSION:-all}"
  PYTORCH_VERSION="${PYTORCH_VERSION:-all}"
  if [ -z "$TAG" ]; then
    DEFAULT_TAG=1
  else
    DEFAULT_TAG=0
  fi

  # Parse JSON and create .txt files
  python parse_json.py -p "$PYTHON_VERSION" -v "$PYTORCH_VERSION"

  # Iterate over .txt files
  txt_files=$(find "$ABS_DIR_PATH" -type f -name "*.txt")
  for file in $txt_files; do
    file_content=$(tr '\n' ' ' < "$file")
    read -r PYTORCH_VERSION PYTHON_VERSION <<< "$file_content"

    # Map PYTORCH_VERSION to PYTORCH_FULL_VERSION
    case $PYTORCH_VERSION in
      "1.9") PYTORCH_FULL_VERSION="1.9.0" ;;
      "1.13") PYTORCH_FULL_VERSION="1.13.1" ;;
      *) echo "Unknown PYTORCH_FULL_VERSION"; exit 1 ;;
    esac

    # Set TAG if not provided
    if [[ $DEFAULT_TAG == 1 ]]; then TAG="$MMCV_BRANCH-pytorch$PYTORCH_VERSION-$PYTHON_VERSION"; fi

    # Set PY_VERSION and PY_SUFFIX
    PY_VERSION="${PYTHON_VERSION//./}"
    PY_SUFFIX="py${PY_VERSION}"

    # Print build arguments
    print_build_args
    # Build the wheel
    build_wheel_func
    # Copy to local
    copy2local_func
  done

  # Clean up .txt files
  rm -f "$ABS_DIR_PATH"/*.txt
}

print_build_args(){
  echo "========= print all build args ========="
  echo "torch_version is: ${PYTORCH_VERSION}"
  echo "torch_full_version is: ${PYTORCH_FULL_VERSION}"
  echo "catch_version is: ${CATCH_VERSION}"
  echo "python_version is: ${PYTHON_VERSION}"
  echo "mmcv_branch is: ${MMCV_BRANCH}"
  echo "py_suffix is: ${PY_SUFFIX}"
  echo "py_version is: ${PY_VERSION}"
  echo "image tag is: ${TAG}"
  echo "========================================"
}

# build wheel from Dockerfile
build_wheel_func(){
  build_wheel_cmd="docker build --shm-size 100G --cpuset-cpus="0-64"           \
                   --memory-swap -1 --no-cache --network=host                  \
                   --build-arg torch_version=${PYTORCH_VERSION}                \
		   --build-arg torch_full_version=${PYTORCH_FULL_VERSION}      \
                   --build-arg catch_version=${CATCH_VERSION}                  \
                   --build-arg python_version=${PYTHON_VERSION}                \
		   --build-arg mmcv_branch=${MMCV_BRANCH}                      \
                   --build-arg py_suffix=${PY_SUFFIX}                          \
		   --build-arg py_version=${PY_VERSION}                        \
                   -t ${IMAGE_WHEEL_NAME}:${TAG} --file ${DOCKER_FILE} ."
  echo "build_wheel_func command: "$build_wheel_cmd
  eval $build_wheel_cmd
}

# pack src func in host
pack_src_func(){
  MMCV_PACKAGE="cambricon_mmcv"
  version=$(grep -o '"version": "[^"]*' "$ABS_DIR_PATH/build.property" | awk -F'"' '{print $4}')
  mmcv_version=$(grep -o '"official_version": "[^"]*' "$ABS_DIR_PATH/build.property" | awk -F'"' '{print $4}')
  rm -rf ${MMCV_PACKAGE}
  mkdir ${MMCV_PACKAGE}
  pushd ${MMCV_PACKAGE}

  # step 1: git clone source codes
  git clone http://gitlab.software.cambricon.com/neuware/oss/openmmlab/mmcv.git -b ${MMCV_BRANCH} --depth 1

  # step 2: remove docs in mmcv
  rm -rf ${PYTORCH_SRC_PACKAGE}/mmcv/docs

  # step 3: remove git/jenkins/other info
  chmod +x ./mmcv/packaging/scripts/mmcv_remove_files.sh

  popd # to cambricon_mmcv/../

  # step 4: pack
  pack_src_cmd="tar cfz 'Cambricon-MMCV${mmcv_version}-${version}.tar.gz' ${MMCV_PACKAGE}"
  eval $pack_src_cmd

  # step 5: remove
  rm -rf $MMCV_PACKAGE
}

copy2local_func() {
  # Container and image names
  current_time=$(date +"%Y-%m-%d-%H-%M-%S")
  container_name="dummy-${current_time}"
  image_name="${IMAGE_WHEEL_NAME}:${TAG}"

  # Create Docker container
  if docker create -it --name "$container_name" "$image_name" /bin/bash; then
    echo "Docker container created successfully."
  else
    echo "Error: Failed to create Docker container."
    return 1
  fi

  # Copy files from Docker container to local machine
  if docker cp "$container_name:/wheel/" .; then
    echo "Files copied from Docker container to local machine successfully."
  else
    echo "Error: Failed to copy files from Docker container."
    return 1
  fi

  # Destroy Docker container
  if docker rm -f "$container_name"; then
    echo "Docker container destroyed successfully."
  else
    echo "Error: Failed to destroy Docker container."
    return 1
  fi

  # Move the copied files to a version-specific directory
  mv ./wheel/ "./wheel_$PY_VERSION-pytorch${PYTORCH_VERSION}"

  # Remove image
  if docker rmi -f "$image_name"; then
    echo "Docker image destroyed successfully."
  else
    echo "Error: Failed to remove Docker image."
    return 1
  fi
}

function build_docs() {
  doc_build_image="yellow.hub.cambricon.com/toolrd/makedocs:latest"
  docs_output="${PWD}/docs_output/"
  mkdir -p "${docs_output}"

  # Run documentation build in Docker container
  bash run_in_docker.sh ${doc_build_image} bash build.sh ${docs_output} ${MMCV_BRANCH}

  # Check for build success
  if [ $? -ne 0 ]; then
    echo "Error: Build mmcv docs failed."
    exit 1
  fi
  
  # Move files and clean up 
  mv ${docs_output}/docs/*.zip ${docs_output}
  mv ${docs_output}/docs/*.pdf ${docs_output}
  rm -rf ${docs_output}/docs
}


if [ -z "$MMCV_BRANCH" ]; then
  echo "MMCV branch must be specified using -m"
  exit
fi

if [[ $RELEASE_TYPE == "wheel" ]]; then
  echo "=== BUILD WHEEL ==="
  build_all_wheels_func
elif [[ ${RELEASE_TYPE} == "src" ]]; then
  echo "=== RELEASE SRC ==="
  pack_src_func
elif [[ ${RELEASE_TYPE} == "doc" ]]; then
  echo "=== BUILD DOC ==="
  build_docs
elif [[ ${RELEASE_TYPE} == "all" ]]; then
  echo "=== BUILD ALL ==="
  build_all_wheels_func
  pack_src_func
  build_docs
else
  echo "unrecognized RELEASE_TYPE: "$RELEASE_TYPE
fi
