__precompile__(true)

module StandardMarketData

using Printf, Dates, Distributed, Mmap, Statistics, Random
using ProgressMeter, Parameters, Glob, MacroTools
using PyCall, HDF5, BSON, BSONMmap
using PandasLite, PyCallUtils, HDF5Utils
using BlockArrays: _BlockArray
using PandasLite: StringRange
import StatsBase

export Data, loaddata, savedata, reloaddata
export nfeats, ncodes, nticks, ndays, nticksperday
export discretize!, discretize, undiscretize, period, downsample, featnames
export getfeat, getfeats, @uncol, dropfeats, keepfeats, getcats, keepcats
export datespan, firstdate, lastdate, setcomm, setpool
export epochsof, datetimesof, datesof, codesof, parsefreq
export concat, pivot, rolldata, to_df, to_data, sourceof, findsnap, splitday
export unix2date, unix2time, unix2hour, unix2str8, unix2str6, unix2int
export str2date, str2datetime, str2unix, int2unix
export to_dict, to_struct, idxmap, pct_change
export to_category, from_category
export sortedunique, sortednunique, nunique
export fillnan, fillnan!
export extract_talib_feats, extract_tsfresh_feats
export SMD, ⧶, @staticvar, @staticdef

export creshape, rreshape, csize, rsize, ccount, rcount
export cview, rview, cget, rget, cset!, rset!
export concatenate, cconcatenate, rconcatenate
export unsqueeze, stack, unstack, cstack, rstack
export colvec, rowvec, indbatch, minibatch, splat

const SMD = StandardMarketData

include("data.jl")
include("util.jl")
include("feature.jl")
include("discretize.jl")
include("array.jl")

end