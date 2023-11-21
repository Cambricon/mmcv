#!/bin/bash
set -x

version=$1
DEST_HOST=$2
passwd=$3
RELEASE_TYPE=$4
mmcv_version=$5

daily_path="http://daily.software.cambricon.com/daily/mmcv"
daily_dir="/data/daily/mmcv"
release_dir="/data/release/mmcv"

branch=`git symbolic-ref --short -q HEAD`
build_version=$1-mmcv$5-`date +%Y%m%d_%H%M%S`-`git rev-parse --short HEAD`

sample_code_path="${daily_dir}/src"
docs_path="${daily_dir}/docs"

function create_remote_wheel_dir() {
  wheels_path="${daily_dir}/wheel/${2}/python${1}"
  echo "#!/usr/bin/expect -f" >> expect_script
  echo "
  set prompt {[#>$] }
  spawn ssh ${DEST_HOST}
  expect {
  \"yes/no\" { send \"yes\r\"; exp_continue}
  \"*password:\" { send \"${passwd}\r\" }
  }
  expect \"\$prompt\"
  send \"mkdir -p ${wheels_path}/${branch}/${build_version}\n\"
  send \"rm -rf ${wheels_path}/${branch}/latest\n\"
  send \"ln -s ${wheels_path}/${branch}/${build_version} ${wheels_path}/${branch}/latest\n\"
  expect eof " >> expect_script
  expect expect_script
  rm -rf expect_script
}

function create_remote_sample_code_dir() {
  echo "#!/usr/bin/expect -f" >> expect_script
  echo "
  set prompt {[#>$] }
  spawn ssh ${DEST_HOST}
  expect {
  \"yes/no\" { send \"yes\r\"; exp_continue}
  \"*password:\" { send \"${passwd}\r\" }
  }
  expect \"\$prompt\"
  send \"mkdir -p ${sample_code_path}/${branch}/${build_version}\n\"
  send \"rm -rf ${sample_code_path}/${branch}/latest\n\"
  send \"ln -s ${sample_code_path}/${branch}/${build_version} ${sample_code_path}/${branch}/latest\n\"
  expect eof " >> expect_script
  expect expect_script
  rm -rf expect_script
}

function create_remote_docs_dir() {
  echo "#!/usr/bin/expect -f" >> expect_script
  echo "
  set prompt {[#>$] }
  spawn ssh ${DEST_HOST}
  expect {
  \"yes/no\" { send \"yes\r\"; exp_continue}
  \"*password:\" { send \"${passwd}\r\" }
  }
  expect \"\$prompt\"
  send \"mkdir -p ${docs_path}/${branch}/${build_version}\n\"
  send \"rm -rf ${docs_path}/${branch}/latest\n\"
  send \"ln -s ${docs_path}/${branch}/${build_version} ${docs_path}/${branch}/latest\n\"
  expect eof " >> expect_script
  expect expect_script
  rm -rf expect_script
}

function funcSum() {
  WHEEL_NAME=$1
  md5sum ${WHEEL_NAME} > ${WHEEL_NAME}.md5sum
  sha256sum ${WHEEL_NAME} > ${WHEEL_NAME}.sha256sum
}

function upload_wheels() {
  echo "uploading wheels..."
  whl_file=$(find "./wheel_${1}/" -type f -name "*.whl")
  filename=$(basename "$whl_file")
  echo "file=${filename}" > version.txt
  
  funcSum $whl_file
  sshpass -p ${passwd} scp -r ./wheel_${1}/* ${DEST_HOST}:${wheels_path}/${branch}/${build_version}/
  if [ $? != 0 ];then
    echo "upload docs faild."
    exit -1
  fi
  sshpass -p ${passwd} scp -r version.txt ${DEST_HOST}:${wheels_path}/${branch}/${build_version}/
  echo "done"
}

function upload_docs() {
  echo "uploading docs..."

  sshpass -p ${passwd} scp -r docs_output/* ${DEST_HOST}:${docs_path}/${branch}/${build_version}/
  if [ $? != 0 ];then
    echo "upload docs faild."
    exit -1
  fi
  echo "done"
}

function upload_src() {
  echo "uploading src..."

  target_src_pkg="Cambricon-MMCV${mmcv_version}-${version}.tar.gz"
  echo "file=${target_src_pkg}" > version.txt

  sshpass -p ${passwd} scp -r ${target_src_pkg} ${DEST_HOST}:${sample_code_path}/${branch}/${build_version}/
  if [ $? != 0 ];then
    echo "upload src faild"
    exit -1
  fi
  sshpass -p ${passwd} scp -r version.txt ${DEST_HOST}:${sample_code_path}/${branch}/${build_version}/
  echo "done"
}

if [[ $RELEASE_TYPE == "wheel" ]]; then
  echo "=== UPLOAD WHEEL ==="
  # Iterate over directories starting with "wheel_"
  for dir in wheel_*; do
      if [ -d "$dir" ]; then  # Check if it's a directory
          suffix=${dir#*_}    # Extract the suffix after "_"
	  versions_inf=${suffix:0:1}.${suffix:1}
	  OLD_IFS=$IFS
	  IFS='-'
	  read -r python_version pytorch_version <<< "$versions_inf"
	  IFS=$OLD_IFS
          create_remote_wheel_dir $python_version $pytorch_version
	  upload_wheels $suffix
      fi
  done
elif [[ ${RELEASE_TYPE} == "src" ]]; then
  echo "=== UPLOAD RELEASE SRC ==="
  create_remote_sample_code_dir
  upload_src
elif [[ ${RELEASE_TYPE} == "doc" ]]; then
  echo "=== UPLOAD DOC ==="
  create_remote_docs_dir
  upload_docs
elif [[ ${RELEASE_TYPE} == "all" ]]; then
  echo "=== UPLOAD ALL ==="
  for dir in wheel_*; do
      if [ -d "$dir" ]; then  # Check if it's a directory
          suffix=${dir#*_}    # Extract the suffix after "_"
	  versions_inf=${suffix:0:1}.${suffix:1}
	  OLD_IFS=$IFS
	  IFS='-'
	  read -r python_version pytorch_version <<< "$versions_inf"
	  IFS=$OLD_IFS
          create_remote_wheel_dir $python_version $pytorch_version
	  upload_wheels $suffix
      fi
  done
  create_remote_sample_code_dir
  upload_src
  create_remote_docs_dir
  upload_docs
else
  echo "unrecognized RELEASE_TYPE: "$RELEASE_TYPE
fi
