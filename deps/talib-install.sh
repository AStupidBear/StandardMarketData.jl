#!/bin/bash
TALIB=$JULIA_DEPOT_PATH/talib
if [ ! -d $TALIB ]; then
    url=http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
    cd /tmp/ && wget $url
    tar xvzf ta-lib-0.4.0-src.tar.gz
    cd ta-lib && ./configure --prefix=$TALIB
    make && make install
fi