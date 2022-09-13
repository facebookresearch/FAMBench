#!/bin/bash
if [ $# -ne 2 ]; then
    echo "Usage: bash $(basename $0) <FILEID> <FILENAME>"
    exit -1
fi

FILEID=$1
FILENAME=$2

DL_LINK="https://docs.google.com/uc?export=download&id=${FILEID}"

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate ${DL_LINK} -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${FILEID}" -O ${FILENAME} && rm -rf /tmp/cookies.txt
