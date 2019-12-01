__precompile__(true)

module StandardMarketData

using Printf, Dates, Distributed, Mmap
using BlockArrays: _BlockArray
using PyCall, PandasLite, PyCallUtils
using PandasLite: StringRange, PandasWrapped
import StatsBase
using ProgressMeter, Parameters

export roll, @roll

export Data, loaddata, savedata, reloaddata, initdata
export nfeas, nstocks, nticks, ndays, nticksperday, period, isstocks
export prune, rescale!, rescale, downsample, colnames
export col, cols, @uncol, dropcols, keepcols, categories, keepcats
export datespan, firstdate, lastdate, getlabel, setcomm, setpool
export epochsof, timesof, codesof, datesof, normalmask, parsefreq, concat, pivot
export int2str, str2int, flt2str, str2flt, normalize_code
export unix2date, unix2time, unix2str8, unix2str6, str2date, str2datetime, str2unix, unix2int, int2unix
export to_df, to_data, metacols, feacols, split_metafea, lngstconsec

include("data.jl")
include("util.jl")
include("feature.jl")
include("string.jl")

end