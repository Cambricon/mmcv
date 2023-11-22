#! /bin/bash

make clean
make latexpdf
make html&&zip -qr -P"Cambricon@doc123456" _build/mmcv2.1_release_notes_html.zip _build/html
