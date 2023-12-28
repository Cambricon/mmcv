#!/bin/bash

script_path=`dirname $0`
echo "${script_path}"
pushd $script_path

###Release notes ###
# pushd release_notes
# rm -rf _build
# ./makelatexpdf.sh
# cp _build/latex/Cambricon*.pdf ../
# cp _build/mmcv1.7_release_notes_html.zip ../
# popd

###User guide ###
pushd user_guide
rm -rf _build
./makelatexpdf.sh
cp _build/latex/Cambricon*.pdf ../
cp _build/mmcv1.7_user_guide_html.zip ../
popd

popd

