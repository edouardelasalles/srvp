#!/bin/sh

TARGET_DIR=$1
if [ -z $TARGET_DIR ]
then
    echo "Must specify target directory"
else
    wget -O download_edenton.sh https://raw.githubusercontent.com/edenton/svg/master/data/download_kth.sh
    bash download_edenton.sh $TARGET_DIR
    rm download_edenton.sh
fi
