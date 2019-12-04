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
export getfeat, getfeats, @uncol, dropfeats, keepfeats, getcats, keepcats
export datespan, firstdate, lastdate, getlabel, setcomm, setpool
export epochsof, datetimesof, datesof, codesof, parsefreq
export concat, pivot, roll, to_df, to_data
export unix2date, unix2time, unix2str8, unix2str6, unix2int
export str2date, str2datetime, str2unix, int2unix
export normalize_code, isfutcode, iscommcode, next_tradetime, isholiday
export lngstconsec, reindex_columns, concat_hdfs, concat_txts, catlag
export to_dict, to_struct, parsefreq, sortednunique, idxmap, ⧶
export SMD, talib, tsfresh, to_category, from_category

const SMD = StandardMarketData

include("data.jl")
include("util.jl")
include("feature.jl")

end