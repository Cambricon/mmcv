# Docker images

There exists a `Dockerfile` file to build docker images using manylinux base image to build a wheel

```text
.
|-- README.md
|-- mlu_docker  # save dockerfiles for building mmcv wheel using mlu
|   `-- Dockerfile
`-- scripts  # scripts to help building wheel and copy to local
    `-- build_wheel.sh
```

## Build docker images

example for docker build using provided dockerfile

```bash
docker build --shm-size 100G --cpuset-cpus="0-64" --build-arg arch_list="3.0;5.0" --memory-swap -1 --no-cache \
-t yellow.hub.cambricon.com/mmcv/base/x86_64/mmcv:temp --file packaging/mlu_docker/Dockerfile .
```

The building process may take 10 minutes or more.

## Run images

```bash
docker run -dit --name=temp_mmcv --privileged --network=host yellow.hub.cambricon.com/mmcv/base/x86_64/mmcv:temp /bin/bash
```

the generated mmcv wheel file is under /wheel
