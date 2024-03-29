mutable struct Data{
    S <: AbstractString,
    A <: AbstractArray{T, 3} where T, 
    B <: AbstractMatrix, C <: AbstractMatrix, 
    D <: AbstractMatrix, E <: AbstractMatrix, 
    F<: AbstractMatrix, G <: AbstractMatrix, 
    H <: AbstractMatrix, I <: AbstractMatrix, 
    J <: AbstractMatrix, K <: AbstractMatrix
    }
    特征名::Dict{S, Int}
    特征::A
    涨幅::B
    时间戳::C
    代码::D
    最新价::E
    买1价::F
    卖1价::G
    手续费率::H
    涨停::I
    跌停::J
    交易池::K
end

Base.:(==)(x::Data, y::Data) = all(s -> isequal(getfield(x, s), getfield(y, s)), fieldnames(Data))

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
        code = eltype(data.代码)(code)
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

function ndays(data)
    Δts = Float32[]
    for n in 1:size(data, 1)
        ts = filter(!isna, data.时间戳[n, :])
        !isempty(ts) && return sortednunique(unix2date, ts)
    end
    return 0
end

nticksperday(data) = nticks(data) ÷ ndays(data)

const _edgemap = Dict{String, Vector{Float32}}()

const _sourcemap = Dict{UInt, String}()

sourceof(x) = get(_sourcemap, objectid(x), nothing)

function _loaddata(src; fload = nothing, ti = nothing, tf = nothing, ka...)
    if endswith(src, ".bson")
        data = isnothing(fload) ? bsload(src, Data; ka...) : fload(src; ka...)
    else
        data = isnothing(fload) ? h5load(src, Data; ka...) : fload(src; ka...)
        if "bin_edges" ∈ h5open(names, src)
            bin_edges = h5read(src, "bin_edges")
            for (c, f) in data.特征名
                _edgemap[c] = bin_edges[:, f]
            end
        end
    end
    if isnothing(ti) && isnothing(tf)
        _sourcemap[objectid(data)] = src
        return data
    else
        ti = something(ti, "20000101")
        tf = something(tf, "20501231")
        data′ = @view data[:, ti:tf]
        _sourcemap[objectid(data′)] = src
        return data′
    end
end

function loaddata(srcs::AbstractArray, a...; dims = -1, ka...)
    datas = @showprogress 10 "loaddata..." map(srcs) do src
        _loaddata(src, a...; ka...)
    end
    length(srcs) == 1 && return datas[1]
    concat(filter(!isempty, datas); dims)
end

function loaddata(pattern, a...; ka...)
    if isfile(pattern)
        srcs = [pattern]
    elseif occursin("*", pattern)
        if startswith(pattern, "/")
            srcs = glob(pattern[2:end], "/")
        else
            srcs = glob(pattern)       
        end
    else
        srcs = []
    end
    if isempty(srcs) && !startswith(pattern, "/")
        job = get(ENV, "JOB", expanduser("~/job"))
        srcs = glob(pattern, joinpath(job, "data"))
    end
    isempty(srcs) && error(pattern * " not found")
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

function initdata(dst, eltype_, (F, N, T), feature = nothing)
    isfile(dst) && rm(dst)
    feature = something(feature, string.(1:F))
    h5open(dst, "w") do fid
        g_create(fid, "nonarray")
        @showprogress 10 "initdata..." for s in afieldnames(Data)
            if s == :时间戳
                d_zeros(fid, string(s), Float64, (N, T))
            elseif s == :代码
                fid["代码"] = [MLString{8}(string(n)) for n in 1:N, t in 1:T]
            elseif s == :涨停 || s == :跌停 || s == :交易池
                d_zeros(fid, string(s), UInt8, (N, T))
            elseif s == :特征
                d_zeros(fid, string(s), eltype_, (F, N, T))
            else
                d_zeros(fid, string(s), Float32, (N, T))
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
    @printf(io, "日期范围: %s/%s\t", Dates.format.(datespan(data), dateformat"yymmdd")...)
    @printf(io, "价格范围: %.3g/%.3g\n", extrema(filter(!isna, data.最新价))...)
    @printf(io, "涨幅范围: %.2g/%.2g\n", extrema(filter(!isna, data.涨幅))...)
    compact && return
    header, stats = String[], Array{String}[]
    for f in ("涨幅", "手续费率")
        x = vec(getfield(data, Symbol(f)))
        s = StatsBase.summarystats(x)
        push!(header, f)
        push!(stats, split(string(s), '\n')[2:end-1])
    end
    for (f, i) in data.特征名
        x = filter(z -> !isnan(z) & !isinf(z), data.特征[i, :, :])
        if any(!iszero, x)
            s = StatsBase.summarystats(x)
        else
            s = StatsBase.SummaryStats(zeros(6)..., length(x), 0)
        end
        push!(header, string(f))
        push!(stats, split(string(s), '\n')[2:end-1])
    end
    for (header′, stats′) in zip(Iterators.partition(header, 4), Iterators.partition(stats, 4))
        print_header_stats(io, header′, stats′)
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
            print(io, stats[c][min(r, end)], '\t')
        end
    end
end

function period(data)
    Δts = Float32[]
    for n in 1:size(data, 1)
        ts = filter(!isna, data.时间戳[n, :])
        length(ts) < 2 && continue
        Δt = median(diff(ts))
        2 * length(ts) > size(data, 2) && return Δt
        push!(Δts, Δt)
    end
    isempty(Δts) ? 0f0 : median(Δts)
end

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
    涨幅 = Array(data.涨幅)
    for t′ in 1:(length(ts) - 1)
        t⁻ = t′ == 1 ? 0 : ts[t′ - 1]
        @inbounds for t in (t⁻ + 1):ts[t′], n in 1:N
            data′.涨幅[n, t′] += 涨幅[n, t]
        end
    end
    if length(ts) == 1
        copyto!(data′.涨幅, sum(涨幅, dims = 2))
    end
    !average && return data′
    fill!(data′.特征, 0)
    @showprogress 10 for t′ in 1:(length(ts) - 1)
        t⁻ = t′ == 1 ? 0 : ts[t′ - 1]
        Δt = ts[t′] - t⁻
        @inbounds for t in (t⁻ + 1):ts[t′], n in 1:N, f in 1:F
            data′.特征[f, n, t′] += data.特征[f, n, t] / Δt
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

datespan(data::Data) = (firstdate(data), lastdate(data))

firstdate(data::Data) = unix2date(minimum(t -> ifelse(isna(t), Inf, t), data.时间戳))

lastdate(data::Data) = unix2date(maximum(t -> ifelse(isna(t), -Inf, t), data.时间戳))

pct_change(data::Data, h::String) = pct_change(data, ceil(Int, min(nticks(data), parsefreq(h) / period(data))))

pct_change(data::Data, h::Int) = pct_change(Array(data.涨幅), h)

function pct_change(v, h)
    h = ceil(Int, h)
    N, T = size(v)
    chg = zeros(Float32, N, T)
    r = zeros(Float32, N)
    for t in T:-1:max(2, T - h + 1), n in 1:N
        r[n] += v[n, t]
        chg[n, t - 1] = r[n]
    end
    for t in (T - h):-1:2
        for n in 1:N
            r[n] += v[n, t] - v[n, t + h]
            chg[n, t - 1] = r[n]
        end
    end
    return chg
end

function setcomm(data, c)
    dict = to_dict(data)
    dict[:手续费率] = fill(Float32(c), size(data))
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

function concat(datas; dims = -1)
    fvs = []
    for s in fieldnames(Data)
        xs = getfield.(datas, s)
        x1 = getfield(datas[1], s)
        if isa(x1, AbstractArray)
            d1 = dims > 0 ? dims : (ndims(x1) + 1 + dims)
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

Base.isempty(data::Data) = length(data) == 0 || !any(!isna, data.时间戳)

Base.copy(data::Data) = to_struct(Data, to_dict(data))

pivot(datas::AbstractArray{<:Data}, a...; ka...) = pivot(concat(map(vec, datas)), a...; ka...)

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
    @showprogress 10 "pivot.index" for t in 1:T, n in 1:N
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
            s == :最新价 && Threads.@threads for t′ in 2:T′
                for n′ in 1:N′
                    if iszero(dest[n′, t′])
                        dest[n′, t′] = dest[n′, t′ - 1]
                    end
                end
                next!(p)
            end
            s == :最新价 && Threads.@threads for t′ in (T′ - 1):-1:1
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
    @showprogress 10 "to_df..." for s in afieldnames(Data)
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
        @showprogress 10 "to_data.meta..." for c in metacols(df)
            if c == "时间戳" && df[c].dtype.kind == "M"
                x = df[c].view("int64").div(1e9).values
            elseif c == "代码"
                n = cld(df["代码"].str.len().max(), 8) * 8
                x = from_category(df[c], MLString{n})
            else
                x = df[c].values
            end
            fid[c][:, :] = reshape(x, N, :)
        end
        fid["涨停"][:, end] = 0
        fid["跌停"][:, end] = 0
        Δt = 1024^3 ÷ (4 * F * N)
        rows = LinearIndices((1:N, 1:T))
        @showprogress 10 "to_data.fea..." for t in 1:Δt:T
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

function findsnap(data::Data, hsnap)
    stamps = data.时间戳
    for t in 1:size(stamps, 2)
        zmax = 0.0
        @inbounds @simd for n in 1:size(stamps, 1)
            z = stamps[n, t]
            zmax = ifelse(isna(z), zmax, max(zmax, z))
        end
        unix2hour(zmax) > hsnap && return max(1, t - 1)
    end
    return size(stamps, 2)
end

function splitday(data::Data)
    unix = vec(maximum(fillnan, data.时间戳, dims = 1))
    is = findall(diff(unix .÷ 86400) .>= 1)
    rs = UnitRange.([1; is .+ 1], [is; length(unix)])
    return [view(data, :, r) for r in rs]
end
