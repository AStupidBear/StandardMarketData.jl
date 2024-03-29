using PyCall: python

ENV["TA_LIBRARY_PATH"] = joinpath(DEPOT_PATH[1], "talib/lib")
ENV["TA_INCLUDE_PATH"] = joinpath(DEPOT_PATH[1], "talib/include")
ENV["JULIA_DEPOT_PATH"] = DEPOT_PATH[1]

run(`$python -m pip install --upgrade pip`)
run(`$python -m pip install pandas tsfresh pyarrow`)
if !Sys.iswindows()
    run(`bash talib-install.sh`)
    run(`$python -m pip --no-cache-dir install TA-Lib`)
end
