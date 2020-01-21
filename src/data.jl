mutable struct Data{F <: AbstractArray,
                    R <: AbstractMatrix,
                    C <: AbstractMatrix,
                    L <: AbstractMatrix,
                    C′ <: AbstractMatrix,
                    T <: AbstractMatrix,
                    P <: AbstractMatrix,
                    P′ <: AbstractMatrix}
    特征名::Dict{String, Int}
    特征::F
    涨幅::R
    买手续费率::C
    卖手续费率::C
    涨停::L
    跌停::L
    代码::C′
    时间戳::T
    价格::P
    交易池::P′
end

Base.:(==)(x::Data, y::Data) = all(s -> getfield(x, s) == getfield(y, s), fieldnames(Data))

afieldnames(t) = Symbol[fieldname(t, n) for n in 1:fieldcount(t) if fieldtype(t, n) <: AbstractArray]

featnames(data) = first.(sort(collect(data.特征名), by = last))

mapdata(f, data) = Data(data.特征名, [f(getfield(data, s)) for s in afieldnames(Data)]...)

for f in [:getindex, :view]
    @eval Base.$f(data::Data, is...) = mapdata(data) do x
        ndims(x) == 2 ? $f(x, is...) : $f(x, :, is...)
    end

    @eval function Base.$f(data::Data, ts::StringRange)
        ti = str2unix(ts.start)
        tf = str2unix(ts.stop)
        mask = ti .<= data.时间戳 .<= tf
        ns = findall(dropdims(any, mask, dims = 2))
        ts = findall(dropdims(any, mask, dims = 1))
        $f(data, arr2rng(ns), arr2rng(ts))
    end

    @eval function Base.$f(data::Data, code::String)
        code = MLString{8}(code)
        mask = data.代码 .== code
        ns = findall(vec(any(mask, dims = 2)))
        ts = findall(vec(any(mask, dims = 1)))
        $f(data, arr2rng(ns), arr2rng(ts))
    end
end

Base.maybeview(data::Data, is...) = view(data, is...)

Base.reshape(data::Data, is...) = mapdata(data) do x
    ndims(x) == 2 ? reshape(x, is...) : reshape(x, size(x, 1), is...)
end

Base.vec(data::Data) = reshape(data, 1, length(data))

nfeats(data) = size(data.特征, 1)

ncodes(data) = size(data.涨幅, 1)

nticks(data) = size(data.涨幅, 2)

Base.size(data::Data) = size(data.涨幅)

Base.size(data::Data, dim) = size(data.涨幅, dim)

Base.length(data::Data) = prod(size(data))

Base.lastindex(data::Data, n) = size(data, n)

ndays(data) = sortednunique(unix2date, view(data.时间戳, 1, :))

nticksperday(data) = nticks(data) ÷ ndays(data)

const _edgemap = Dict{String, Vector{Float32}}()

const _sourcemap = Dict{UInt, String}()

sourceof(x) = get(_sourcemap, hash(x), nothing)

function _loaddata(src; mode = "r", ti = nothing, tf = nothing, ka...)
    if endswith(src, ".h5")
        data = h5load(src, Data; mode = mode, ka...)
        if "bin_edges" ∈ h5open(names, src)
            bin_edges = h5read(src, "bin_edges")
            for (c, f) in data.特征名
                _edgemap[c] = bin_edges[:, f]
            end
        end
    elseif endswith(src, ".bson")
        data = bsload(src, Data; ka...)
    end
    if isnothing(ti) && isnothing(tf)
        _sourcemap[hash(data)] = src
        return data
    else
        ti = something(ti, "20000101")
        tf = something(tf, "20501231")
        data′ = @view data[:, ti:tf]
        _sourcemap[hash(data′)] = src
        return data′
    end
end

function loaddata(srcs::AbstractArray, dim = -1; ka...)
    datas = @showprogress "loaddata..." map(srcs) do src
        _loaddata(src; ka...)
    end
    length(srcs) == 1 && return datas[1]
    concat(filter(!isempty, datas), dim)
end

function loaddata(pattern, a...; ka...)
    if isfile(pattern)
        srcs = [pattern]
    else
        srcs = glob(pattern)
    end
    if isempty(srcs)
        root = joinpath(get(ENV, "JOB", ""), "data")
        srcs = glob(pattern, root)
    end
    loaddata(srcs, a...; ka...)
end

function savedata(dst, data)
    if endswith(dst, ".bson")
        bssave(dst, data[:, :])
    elseif endswith(dst, ".h5")
        h5save(dst, data)
    end
    return dst
end

function reloaddata(data)
    dst = @sprintf("data-%s.h5", randstring())
    savedata(dst, data)
    loaddata(dst, mode = "r+")
end

function initdata(dst, eltyp, (F, N, T), feature = nothing)
    isfile(dst) && rm(dst)
    feature = something(feature, string.(1:F))
    h5open(dst, "w", "alignment", (0, 8)) do fid
        g_create(fid, "nonarray")
        @showprogress "initdata..." for s in afieldnames(Data)
            if s == :时间戳
                d_zeros(fid, string(s), Float64, N, T)
            elseif s == :代码
                fid["代码"] = [MLString{8}(string(n)) for n in 1:N, t in 1:T]
            elseif s == :涨停 || s == :跌停 || s == :交易池
                d_zeros(fid, string(s), UInt8, N, T)
            elseif s == :特征
                d_zeros(fid, string(s), eltyp, F, N, T)
            else
                d_zeros(fid, string(s), Float32, N, T)
            end
        end
        fid["交易池"][:, :] = 1
        if !isempty(feature)
            特征名 = Dict(reverse(p) for p in enumerate(feature))
            write_nonarray(fid, "特征名", 特征名)
        end
    end
    return dst
end

function Base.show(io::IO, data::Data)
    isempty(data) && return 
    compact = get(io, :compact, false)
    @printf(io, "特征数: %d\t", nfeats(data))
    @printf(io, "品种数: %d\t", ncodes(data))
    @printf(io, "TK数: %d\n", nticks(data))
    @printf(io, "交易池比例: %.2g\t", mean(data.交易池))
    @printf(io, "涨停比例: %.2g\t", mean(data.涨停))
    @printf(io, "跌停比例: %.2g\n", mean(data.跌停))
    ts = Dates.format.(unix2date.(data.时间戳[[1;end]]), "yymmdd")
    @printf(io, "日期范围: %s/%s\t", ts...)
    @printf(io, "价格范围: %.3g/%.3g\n", extrema(data.价格)...)
    @printf(io, "涨幅范围: %.2g/%.2g\n", extrema(data.涨幅)...)
    compact && return
    header, stats = String[], Array{String}[]
    for f in ["涨幅", "买手续费率", "卖手续费率"]
        x = vec(getfield(data, Symbol(f)))
        s = StatsBase.summarystats(x)
        push!(header, f)
        push!(stats, split(string(s), '\n')[2:end-1])
    end
    for (f, i) in data.特征名
        x = vec(data.特征[i, :, :])
        if any(!iszero, x)
            s = StatsBase.summarystats(x)
        else
            s = StatsBase.SummaryStats(zeros(6)..., length(x), 0)
        end
        push!(header, string(f))
        push!(stats, split(string(s), '\n')[2:end-1])
    end
    for (h, stat) in zip(Iterators.partition(header, 4), Iterators.partition(stats, 4))
        print_header_stats(io, h, stat)
    end
end

Base.show(io::IO, ::MIME"text/plain", data::Data) =
    show(IOContext(io, :compact => true), data)

function print_header_stats(io, header, stats)
    print(io, '\n', '\n')
    for h in header
        print(io, h, ['\t' for i in 1:4]...)
    end
    for r in 1:length(stats[1])
        print(io, '\n')
        for c in 1:length(stats)
            print(io, stats[c][r], '\t')
        end
    end
end

_diff(x) = length(x) > 1 ? diff(x) : [zero(eltype(x))]

period(data) = median(_diff(view(data.时间戳, 1, :)))

downsample(data::Data, freq::String; ka...) =
    downsample(data, round(Int, min(nticks(data), parsefreq(freq) / period(data))); ka...)

function downsample(data::Data, freq::Int; phase = 1, ka...)
    freq <= 1 && return data[:, :]
    downsample(data, phase:freq:nticks(data); ka...)
end

function downsample(data::Data, ts::AbstractArray{Int}; average = false)
    data′ = data[:, ts]
    F, N, T = size(data.特征)
    fill!(data′.涨幅, 0)
    @showprogress for t′ in 1:(length(ts) - 1)
        @inbounds for t in ts[t′]:(ts[t′ + 1] - 1), n in 1:N
            data′.涨幅[n, t′] += data.涨幅[n, t]
        end
    end
    if length(ts) == 1
        copyto!(data′.涨幅, sum(data.涨幅, dims = 2))
    end
    !average && return data′
    data′.特征[:, :, 2:(end - 1)] .= 0
    @showprogress for t′ in 2:(length(ts) - 1)
        freq = ts[t′] - ts[t′ - 1]
        @inbounds for t in (ts[t′ - 1] + 1):ts[t′], n in 1:N, f in 1:F
            data′.特征[f, n, t′] += data.特征[f, n, t] / freq
        end
    end
    return data′
end

function getfeat(data::Data, c; denorm = true, errors = "raise")
    if c in keys(data.特征名)
        f = data.特征名[c]
        x = @view data.特征[f, :, :]
        if haskey(_edgemap, c) && denorm
            undiscretize(x, _edgemap[c])
        end
        return x
    elseif Symbol(c) in fieldnames(Data)
        getfield(data, Symbol(c))
    elseif errors == "ignore"
        zeros(Float32, size(data))
    else
        error("$c not in data or data.特征名")
    end
end

function getfeat(data::Data, r::Regex; ka...)
    cs = getfeats(data, r; ka...)
    !isempty(cs) && return cs[1]
    getfeat(data, string(r); ka...)
end

getfeats(data::Data, cs; ka...) = [getfeat(data, c; ka...) for c in cs]

getfeats(data::Data, r::Regex; ka...) = getfeats(data, filter(c -> occursin(r, c), featnames(data)); ka...)

function dropfeats(data, cs::AbstractArray{Int})
    isempty(cs) && return data
    dict = to_dict(data)
    fs = setdiff(1:nfeats(data), cs)
    dict[:特征名] = idxmap(featnames(data)[fs])
    dict[:特征] = view(data.特征, fs, :, :)
    to_struct(Data, dict)
end

dropfeats(data, cs::Array{<:AbstractString}) = dropfeats(data, [get(data.特征名, c, -1) for c in cs])
dropfeats(data, r::Regex) = dropfeats(data, filter(c -> occursin(r, c), featnames(data)))
keepfeats(data, cs::AbstractArray{Int}) = dropfeats(data, setdiff(1:nfeats(data), cs))
keepfeats(data, cs::Array{<:AbstractString}) = dropfeats(data, setdiff(featnames(data), cs))
keepfeats(data, r::Regex) = keepfeats(data, filter(c -> occursin(r, c), featnames(data)))

getcats(data) = filter(!isempty, unique(String.(first.(split.(featnames(data), ':')))))
keepcats(data, cats) = keepfeats(data, filter(c -> any(cat -> startswith(c, cat) || !occursin(':', c), splat(cats)), featnames(data)))
dropcats(data, cats) = keepcats(data, setdiff(getcats(data), splat(cats)))
dropcats(data, r::Regex) = dropcats(data, filter(c -> occursin(r, c), getcats(data)))
keepcats(data, r::Regex) = keepcats(data, filter(c -> occursin(r, c), getcats(data)))

macro uncol(ex)
    cs, rhs = ex.args
    d, cs = gensym(), isa(cs, Symbol) ? [cs] : cs.args
    kd = [:($c = $getfeat($d, $(string(c)))) for c in cs]
    Expr(:block, :($d = $rhs), kd..., d) |> esc
end

Base.split(data::Data, t::String) = @views data["20010101":t], data[t:"30000101"]

datespan(data::Data) = unix2date.(extrema(data.时间戳))

firstdate(data::Data) = unix2date(minimum(data.时间戳))

lastdate(data::Data) = unix2date(maximum(data.时间戳))

getlabel(data::Data, h::String) = getlabel(data, ceil(Int, min(nticks(data), parsefreq(h) / period(data))))

function getlabel(data::Data, h::Int)
    @unpack 涨幅, 代码 = data
    h > nticks(data) && return 涨幅
    N, T = size(data)
    l = zeros(Float32, N, T)
    r = zeros(Float32, N)
    for t in T:-1:(T - h + 1), n in 1:N
        r[n] += 涨幅[n, t]
        l[n, t] = r[n]
    end
    @showprogress "getlabel..." for t in (T - h):-1:1
        for n in 1:N
            r[n] += 涨幅[n, t] - 涨幅[n, t + h]
            l[n, t] = r[n]
        end
    end
    return l
end

function setcomm(data, c)
    dict = to_dict(data)
    for s in [:买手续费率,  :卖手续费率]
        dict[s] = fill(Float32(c), size(data))
    end
    to_struct(Data, dict)
end

function setpool(data, pool)
    ns = findall(vec(any(isone, pool, dims = 2)))
    data′, pool′ = data[ns, :], pool[ns, :]
    data′.交易池 = isone.(pool′)
    return data′
end

setpool(data, c::Union{String, Regex}) = setpool(data, getfeat(data, c))

function Base.repeat(data::Data, n)
    n == 1 && return data
    fvs = []
    for s in fieldnames(Data)
        x = getfield(data, s)
        if isa(x, AbstractMatrix)
            bsizes = (fill(size(x, 1), n), [size(x, 2)])
            x = _BlockArray(repeat([x], n, 1), bsizes...)
        end
        push!(fvs, x)
    end
    return Data(fvs...)
end

epochsof(data::Data) = sort(unique(data.时间戳))

datetimesof(data::Data) = map(unix2datetime, epochsof(data))

datesof(data::Data) = map(unix2date, epochsof(data))

codesof(data::Data) = sort(unique(data.代码))

normal_mask(data::Data) = @. iszero(data.涨停) | iszero(data.跌停)

function isaligned(datas)
    isempty(datas) && return true
    codes = datas[1].代码[:, 1]
    for data in datas[2:end]
        if data.代码[:, 1] != codes
            return false
        end
    end
    return true
end

function concat(datas, dim)
    if dim == -1 && !isaligned(datas)
        h5 = randstring() * ".h5"
        data, = pivot(datas, h5)
        finalizer(data) do d
            rm(h5, force = true)
        end
        return data
    end
    fvs = []
    for s in fieldnames(Data)
        xs = getfield.(datas, s)
        x1 = getfield(datas[1], s)
        if isa(x1, AbstractArray)
            d1 = dim > 0 ? dim : (ndims(x1) + 1 + dim)
            bsizes = [d == d1 ? size.(xs, d) : [size(x1, d)] for d in 1:ndims(x1)]
            xsizes = [d == d1 ? (:) : 1 for d in 1:ndims(x1)]
            x = _BlockArray(reshape(xs, xsizes...), bsizes...)
        else
            x = x1
        end
        push!(fvs, x)
    end
    return Data(fvs...)
end

Base.isempty(data::Data) = length(data) == 0

Base.copy(data::Data) = to_struct(Data, to_dict(data))

pivot(datas::AbstractArray{<:Data}, a...; ka...) = pivot(concat(map(vec, datas), -1), a...; ka...)

pivot(h5::String, a...; ka...) = pivot(loaddata(h5), a...; ka...)

function pivot(data::Data, dst = "pivot.h5"; meta_only = false)
    epochs = epochsof(data)
    codes = codesof(data)
    F = nfeats(data)
    N, T = size(data)
    N′ = length(codes)
    T′ = length(epochs)
    if meta_only
        F, dict = 1, to_dict(data)
        dict[:特征] = zeros(eltype(data), 1,N, T)
        dict[:特征名] = Dict("dummy" => 1)
        data = to_struct(Data, dict)
    end
    index = fill((0, 0), N, T)
    @unpack 代码, 时间戳 = data
    codemap = Dict(reverse.(enumerate(codes)))
    epochmap = Dict(reverse.(enumerate(epochs)))
    @showprogress "pivot.index" for t in 1:T, n in 1:N
        n′ = codemap[代码[n, t]]
        t′ = epochmap[时间戳[n, t]]
        index[n, t] = (n′, t′)
    end
    data′ = similar(data, (F, N′, T′), dst)
    fill!(data′.涨停, 1)
    fill!(data′.跌停, 1)
    fill!(data′.交易池, 0)
    @inbounds for s in afieldnames(Data)
        src = getfield(data, s)
        dest = getfield(data′, s)
        desc = string("pivot.", s)
        p = Progress(T, desc = desc)
        if s == :特征
            Threads.@threads for t in 1:T
                for n in 1:N
                    n′, t′ = index[n, t]
                    for f in 1:F
                        dest[f, n′, t′] = src[f, n, t]
                    end
                end
                next!(p)
            end
        elseif s == :时间戳
            Threads.@threads for t′ in 1:T′
                for n′ in 1:N′
                    dest[n′, t′] = epochs[t′]
                end
                next!(p)
            end
        elseif s == :代码
            Threads.@threads for t′ in 1:T′
                for n′ in 1:N′
                    dest[n′, t′] = codes[n′]
                end
            end
        else
            Threads.@threads for t in 1:T
                for n in 1:N
                    n′, t′ = index[n, t]
                    dest[n′, t′] = src[n, t]
                end
            end
            s == :价格 && Threads.@threads for t′ in 2:T′
                for n′ in 1:N′
                    if iszero(dest[n′, t′])
                        dest[n′, t′] = dest[n′, t′ - 1]
                    end
                end
                next!(p)
            end
            s == :价格 && Threads.@threads for t′ in (T′ - 1):-1:1
                for n′ in 1:N′
                    if iszero(dest[n′, t′])
                        dest[n′, t′] = dest[n′, t′ + 1]
                    end
                end
                next!(p)
            end
        end
    end
    @. data′.跌停 = ifelse(isone(data′.交易池), data′.跌停, 0f0)
    @. data′.涨停 = ifelse(isone(data′.交易池), data′.涨停, 0f0)
    Mmap.sync!(data′)
    return data′, index
end

function pivot(src::AbstractArray{T, N}, index) where {T, N}
    dims′ = ntuple(d -> maximum(I -> I[d], index), N)
    dest = zeros(T, dims′...)
    for n in 1:length(index)
        dest[index[n]...] = src[n]
    end
    return dest
end

Base.show(io::IO, ::MIME"text/plain", ::Array{<:Data}) = 
    print(io, "Array of Data")

function rolldata(data, rolltrn, rolltst)
    ti, tf = map(unix2date, extrema(data.时间戳))
    Δtb = Day(parsefreq(rolltrn) ÷ 3600 ÷ 24)
    Δtf = Day(parsefreq(rolltst) ÷ 3600 ÷ 24)
    inds = rollindices(ti, tf, Δtb, Δtf)
    @views ((data[t1], data[t2]) for (t1, t2) in inds)
end

function to_df(data::Data; columns = nothing, meta_only = false, cnvtdate = false)
    df = DataFrame()
    @showprogress "to_df..." for s in afieldnames(Data)
        x = vec(getfield(data, s))
        !isnothing(columns) && string(s) ∉ columns && continue
        if s == :代码
            df[s] = to_category(x)
        elseif s == :时间戳
            df[s] = x
            if cnvtdate
                df[s] = df[s].mul(1e9).astype("datetime64[ns]")
            end
        elseif s == :特征 && !meta_only
            df = pdhcat(df, pd.DataFrame(pymat(data.特征), columns = featnames(data)))
        else
            df[s] = x
        end
    end
    return df
end

function to_data(df, dst; ka...)
    try
        _to_data(df, dst; ka...)
    catch e
        rm(dst, force = true)
        throw(e)
    end
end

function _to_data(df, dst; ncode = 0)
    isempty(df) && return
    F = length(featcols(df))
    N = ncode == 0 ? df["代码"].nunique() : ncode
    T = length(df) ÷ N
    @assert N * T == length(df)
    dst = initdata(dst, Float32, (F, N, T), featcols(df))
    h5open(dst, "r+") do fid
        @showprogress "to_data.meta..." for c in metacols(df)
            if c == "时间戳" && df[c].dtype.kind == "M"
                x = df[c].astype("int").div(1e9).values
            elseif c == "代码"
                x = from_category(df[c])
            else
                x = df[c].values
            end
            fid[c][:, :] = reshape(x, N, :)
        end
        fid["涨停"][:, end] = 0
        fid["跌停"][:, end] = 0
        Δt = 1024^3 ÷ (4 * F * N)
        rows = LinearIndices((1:N, 1:T))
        @showprogress "to_data.fea..." for t in 1:Δt:T
            ti, tf = t, min(t + Δt - 1, T)
            ri, rf = rows[1, ti], rows[end, tf]
            slc = df.loc[(ri - 1):(rf - 2), featcols(df)]
            slc = slc.astype("float32").T.values
            x = reshape(slc, F, N, :)
            fid["特征"][:, :, ti:tf] = x
        end
    end
    @eval GC.gc()
    return dst
end

featcols(df) = sort!(setdiff(df.columns, string.(afieldnames(Data))))

metacols(df) = sort!(setdiff(df.columns, featcols(df)))

split_metafeat(df) = df[metacols(df)], df[featcols(df)]

function isdatafile(h5)
    try
        loaddata(h5)
        return true
    catch e
        return false
    end
end

Base.eltype(data::Data) = eltype(data.特征)

function Base.similar(data::Data, dims, dst = randstring())
    initdata(dst, eltype(data), dims, featnames(data))
    loaddata(dst, mode = "r+")
end

function Mmap.sync!(data::Data)
    for s in afieldnames(Data)
        Mmap.sync!(getfield(data, s))
    end
end

# function reweight_pnl(data)
#     date = lastdayofmonth.(unix2datetime.(data.时间戳))
#     # pnl = get(记忆收益率, hash(data.时间戳)) do
#     #     abs.(getlabel(data, "4H"))
#     # end
#     pnl = abs.(getlabel(data, "4H"))
#     df = DataFrame("date" => vec(date), "pnl" => vec(pnl))
#     df.set_index("date", inplace = true)
#     df["pnl_avg"] = df["pnl"].groupby("date").mean()
#     w = (1 / df["pnl_avg"].abs()).values
#     clamp!(w, quantile(w, (0.05, 0.95))...)
#     reshape(divavg!(w), size(data))
# end

# function reweight_time(data, scale = "2Y")
#     ts = data.时间戳
#     λ = 1 / parsefreq(scale)
#     ti, tf = extrema(ts)
#     divavg!(@. exp(λ * (ts - tf)))
# end

# reweight(data) = reweight_time(data) .* reweight_pnl(data)