__precompile__(true)

module StandardMarketData

using Printf, Dates, Distributed, Mmap, Statistics, Random
using ProgressMeter, Parameters, Glob, PyCall, HDF5, BSON
using BSONMmap, PandasLite, PyCallUtils, HDF5Utils
using BlockArrays: _BlockArray
using PandasLite: StringRange
import StatsBase

export Data, loaddata, savedata, reloaddata, initdata
export nfeats, ncodes, nticks, ndays, nticksperday, period
export rescale!, rescale, downsample, featnames
export column, columns, @uncol, dropcols, keepcols, categories, keepcats
export datespan, firstdate, lastdate, getlabel, setcomm, setpool
export epochsof, datetimesof, codesof, datesof, parsefreq, concat, pivot
export unix2date, unix2time, unix2str8, unix2str6, unix2int
export str2date, str2datetime, str2unix, int2unix
export roll, to_df, to_data, normalize_code, sortednunique

export SMD
const SMD = StandardMarketData

include("data.jl")
include("util.jl")
include("feature.jl")

end