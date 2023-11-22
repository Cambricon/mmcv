#! /bin/bash

make clean
make latexpdf
make html&&zip -qr -P"Cambricon@doc123456" _build/mmcv1.7_user_guide_html.zip _build/html
