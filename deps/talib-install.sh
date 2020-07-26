#!/bin/bash
TALIB=$JULIA_DEPOT_PATH/talib
if [ ! -d $TALIB ]; then
    url=http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
    cd /tmp/ && wget -O ta-lib.tar.gz $url && tar xvzf ta-lib.tar.gz
    if [ "$(arch)" == aarch64 ]; then
        cd ta-lib && ./configure --prefix=$TALIB --build=arm
    else
        cd ta-lib && ./configure --prefix=$TALIB
    fi
    make && make install
fi