using PyCall: python

ENV["TA_LIBRARY_PATH"] = joinpath(DEPOT_PATH[1], "talib/lib")
ENV["TA_INCLUDE_PATH"] = joinpath(DEPOT_PATH[1], "talib/include")
ENV["JULIA_DEPOT_PATH"] = DEPOT_PATH[1]

run(`bash talib-install.sh`)
run(`$python -m pip install tsfresh chinesecalendar`)
run(`$python -m pip --no-cache-dir install TA-Lib`)