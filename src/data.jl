mutable struct Data{A <: AbstractMatrix{Float32},
                    B <: AbstractMatrix{Float32},
                    C <: AbstractMatrix{Float32},
                    D <: AbstractMatrix{Float64},
                    E <: AbstractArray{Float32, 3},
                    F <: AbstractMatrix{Float32},
                    G <: AbstractMatrix{Float32},
                    H <: AbstractMatrix{Float32}}
    特征名::Dict{String, Int}
    特征::A
    涨幅::B
    买手续费率::C
    卖手续费率::C
    涨停::D
    跌停::D
    代码::E
    时间戳::F
    价格::G
    交易池::H
end

afieldnames(t) = Symbol[fieldname(t, n) for n in 1:fieldcount(t) if fieldtype(t, n) <: AbstractArray]

featnames(data) = first.(sort(collect(data.特征名), by = last))

mapdata(f, data) = Data(data.特征名, [f(getfield(data, s)) for s in afieldnames(Data)]...)

time_mask(data, rng) = str2unix(rng.start) .<= data.时间戳[1, :] .<= str2unix(rng.stop)

Base.getindex(data::Data, is...) = mapdata(data) do x
    ndims(x) == 2 ? x[is...] : x[:, is...]
end

Base.getindex(data::Data, i1, i2::StringRange) = data[i1, time_mask(data, i2)]

Base.getindex(data::Data, code::String) = data[:, vec(data.代码) .== code]

Base.view(data::Data, is...) = mapdata(data) do x
    ndims(x) == 2 ? view(x, is...) : view(x, :, is...)
end

Base.view(data::Data, i1, i2::StringRange) = view(data, i1, time_mask(data, i2))

Base.maybeview(data::Data, is...) = view(data, is...)

Base.reshape(data::Data, is...) = mapdata(data) do x
    ndims(x) == 2 ? reshape(x, is...) : reshape(x, size(x, 1), is...)
end

Base.vec(data::Data) = reshape(data, 1, length(data))

nfeats(data) = size(data.特征, 1)

ncodes(data) = size(data.涨幅, 2)

nticks(data) = size(data.涨幅, 3)

Base.size(data::Data) = size(data.涨幅)

Base.size(data::Data, dim) = size(data.涨幅, dim)

Base.length(data::Data) = prod(size(data))

Base.lastindex(data::Data, n) = size(data, n)

ndays(data) = sortednunique(unix2date, view(data.时间戳, 1, :))

nticksperday(data) = nticks(data) ÷ ndays(data)

const _nrmcoefmap = Dict{String, NTuple{4, Float32}}()

function _loaddata(src; mode = "r", ti = nothing, tf = nothing, ka...)
    if endswith(src, ".h5")
        data = h5load(src, Data; mode = mode, ka...)
        coeff_names = filter(h5open(names, src)) do c
            occursin(r"norm|nrm|归一", c)
        end
        if length(coeff_names) == 1
            coeff = h5read(src, coeff_names[1])
            for (c, f) in data.特征名
                _nrmcoefmap[c] = tuple(coeff[:, f]...)
            end
        end
    elseif endswith(src, ".bson")
        data = bsload(src, Data; ka...)
    end
    isnothing(ti) && isnothing(tf) && return data
    ti = something(ti, "20000101")
    tf = something(tf, "20501231")
    @view data[:, ti:tf]
end

function loaddata(srcs::AbstractArray; dim = -1, ka...)
    datas = @showprogress "loaddata..." map(srcs) do src
        _loaddata(src; ka...)
    end
    length(srcs) == 1 && return datas[1]
    concat(filter(!isempty, datas), dim)
end

function loaddata(pattern; ka...)
    if isfile(pattern)
        srcs = [pattern]
    else
        srcs = glob(pattern)
    end
    if isempty(srcs)
        root = joinpath(get(ENV, "JOB", ""), "data")
        srcs = glob(pattern, root)
    end
    loaddata(srcs; ka...)
end

function savedata(dst, data)
    if endswith(dst, ".bson")
        bssave(dst, data[:, :])
    elseif endswith(dst, ".h5")
        h5save(dst, data)
    end
end

function reloaddata(data)
    dst = @sprintf("data-%s.h5", randstring())
    savedata(dst, data)
    loaddata(dst, mode = "r+")
end

function initdata(dst, F, N, T; columns = [])
    isfile(dst) && rm(dst)
    h5open(dst, "w") do fid
        g_create(fid, "nonarray")
        @showprogress "initdata..." for s in afieldnames(Data)
            if s == :时间戳
                d_zeros(fid, string(s), Float64, N, T)
            elseif s == :特征
                d_zeros(fid, string(s), Float32, F, N, T)
            else
                d_zeros(fid, string(s), Float32, N, T)
            end
        end
        fid["代码"][:, :] .= MLString{8}.(string.(1:N))
        fid["交易池"][:, :] .= 1
        if !isempty(columns)
            特征名 = Dict(reverse(p) for p in enumerate(columns))
            write_nonarray(fid, "特征名", 特征名)
        end
    end
    return dst
end

function Base.show(io::IO, data::Data)
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

function rescale!(srcs; fillnan = true, ignored_columns = [])
    isa(srcs, AbstractArray) || (srcs = [srcs])
    datas = loaddata.(srcs, mode = "r+")
    F, N, T = size(datas[1].特征)
    nrmcoef = zeros(Float32, 4, F)
    ignored_feas = [get(datas[1].特征名, c, c) for c in ignored_columns]
    @showprogress "computing nrmcoef..." for f in 1:F
        if f ∈ ignored_feas
            nrmcoef[:, f] = [-Inf32, Inf32, 0f0, 1f0]
        else
            ts = 1:max(1, T ÷ 10^6):T
            y = filter(!isnan, datas[1].特征[f, :, ts])
            if !isempty(y)
                θd, θu = quantile(y, [0.01f0, 0.99f0])
                clamp!(y, θd, θu)
                μ, σ = mean(y), std(y)
                σ = ifelse(isnan(σ), 1f0, σ)
                nrmcoef[:, f] = [θd, θu, μ, σ]
            else
                nrmcoef[:, f] = [0f0, 0f0, 0f0, 1f0]
            end
        end
    end
    for (src, data) in zip(srcs, datas)
        name = @sprintf("normalizing %s...", basename(src))
        F, N, T = size(data.特征)
        @showprogress name for t in 1:T
            x = data.特征[:, :, t]
            for n in 1:N, f in 1:F
                xfn = clamp(x[f, n], nrmcoef[1, f], nrmcoef[2, f])
                xfn = ifelse(isnan(xfn) && fillnan, nrmcoef[3, f], xfn)
                x[f, n] = (xfn - nrmcoef[3, f]) ⧶ nrmcoef[4, f]
            end
            data.特征[:, :, t] = x
        end
        Mmap.sync!(data.特征)
        h5write(src, "归一系数", nrmcoef)
    end
end

function rescale(h5)
    h5′ = replace(h5, ".h5" => "_NRM.h5")
    run(`cp $h5 $h5′`)
    rescale!(h5′)
end

period(data) = nticks(data) > 1 ? median(diff(view(data.日期, 1, :))) : 1

downsample(data::Data, freq::String; ka...) =
    downsample(data, round(Int, parsefreq(freq) / period(data)); ka...)

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

function column(data::Data, c; denorm = true, errors = "raise")
    if c in keys(data.特征名)
        f = data.特征名[c]
        x = @view data.特征[f, :, :]
        if haskey(_nrmcoefmap, c) && denorm
            θd, θu, μ, σ = _nrmcoefmap[c]
            x = x .* σ .+ μ
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

function column(data::Data, r::Regex; ka...)
    cs = columns(data, r; ka...)
    !isempty(cs) && return cs[1]
    column(data, string(r); ka...)
end

columns(data::Data, cs; ka...) = [column(data, c; ka...) for c in cs]

columns(data::Data, r::Regex; ka...) = columns(data, filter(c -> occursin(r, c), featnames(data)); ka...)

function dropcols(data, cs::AbstractArray{Int})
    isempty(cs) && return data
    dict = to_dict(data)
    fs = setdiff(1:nfeats(data), cs)
    dict["特征名"] = idxmap(featnames(data)[fs])
    dict["特征"] = view(data.特征, fs, :, :)
    to_struct(dict, Data)
end

dropcols(data, cs::Array{<:AbstractString}) = dropcols(data, [get(data.特征名, c, -1) for c in cs])
dropcols(data, r::Regex) = dropcols(data, filter(c -> occursin(r, c), featnames(data)))
keepcols(data, cs::AbstractArray{Int}) = dropcols(data, setdiff(1:nfeats(data), cs))
keepcols(data, cs::Array{<:AbstractString}) = dropcols(data, setdiff(featnames(data), cs))
keepcols(data, r::Regex) = keepcols(data, filter(c -> occursin(r, c), featnames(data)))

categories(data) = filter(!isempty, unique(String.(first.(split.(featnames(data), ':')))))
keepcats(data, cats) = keepcols(data, filter(c -> any(cat -> startswith(c, cat) || !occursin(':', c), splat(cats)), featnames(data)))
dropcats(data, cats) = keepcats(data, setdiff(categories(data), splat(cats)))
dropcats(data, r::Regex) = dropcats(data, filter(c -> occursin(r, c), categories(data)))
keepcats(data, r::Regex) = keepcats(data, filter(c -> occursin(r, c), categories(data)))

macro uncol(ex)
    cs, rhs = ex.args
    d, cs = gensym(), isa(cs, Symbol) ? [cs] : cs.args
    kd = [:($c = $column($d, $(string(c)))) for c in cs]
    Expr(:block, :($d = $rhs), kd..., d) |> esc
end

Base.split(data::Data, t::String) = split(data, str2unix(t))

function Base.split(data::Data, t)
    t = findfirst(x -> t <= x, data.时间戳[1, :])
    @views data[:, 1:(t - 1)], data[:, t:end]
end

datespan(data::Data) = unix2date.(extrema(data.时间戳))

firstdate(data::Data) = unix2date(minimum(data.时间戳))

lastdate(data::Data) = unix2date(maximum(data.时间戳))

getlabel(data::Data, h::String) = getlabel(data, ceil(Int, parsefreq(h) / period(data)))

function getlabel(data::Data, h::Int)
    h > nticks(data) && return data.涨幅
    @unpack 涨幅, 代码 = data
    N, T = size(data)
    l = zeros(Float32, N, T)
    r = zeros(Float32, N)
    for t in T:-1:(T - h + 1), n in 1:N
        r[n] += 涨幅[n, t]
        l[n, t] = r[n] / (T - t + 1)
    end
    @showprogress "getlabel..." for t in (T - h):-1:1, n in 1:N
        if 代码[n, t] != 代码[n, t + 1]
            r[n] = 涨幅[n, t]
        else
            r[n] += 涨幅[n, t] - 涨幅[n, t + h]
        end
        l[n, t] = r[n] / h
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

setpool(data, c::Union{String, Regex}) = setpool(data, column(data, c))

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

epochsof(data::Data) = unique(data.时间戳)

datetimesof(data::Data) = map(unix2datetime, epochsof(data))

datesof(data::Data) = map(unix2date, epochsof(data))

codesof(data::Data) = unique(data.代码)

normal_mask(data::Data) = @. iszero(data.涨停) | iszero(data.跌停)

function concat(datas, dim)
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

pivot(datas::AbstractArray{<:Data}) = concat(map(vec, datas), -1)

function pivot(data::Data; dst = "pivot.h5")
    data = vec(data)
    @unpack 代码, 时间戳 = data
    codes = sort(unique(代码))
    dates = sort(unique(时间戳))
    multidx = pd.MultiIndex.from_product((dates, codes))
    df = DataFrame("code" => vec(代码), "date" => vec(时间戳))
    df = df.reset_index().set_index(["date", "code"]).reindex(multidx)
    df["index"] = df["index"].fillna(-1).astype("int") + 1
    F, N, T = nfeats(data), length(codes), length(dates)
    index = reshape(df["index"].values, N, T)
    initdata(dst, F, N, T, columns = featnames(data))
    data′ = loaddata(dst, mode = "r+")
    fill!(data′.涨停, 1)
    fill!(data′.跌停, 1)
    fill!(data′.交易池, 0)
    for s in fieldnames(Data)
        typ = fieldtype(Data, s)
        typ <: AbstractArray || continue
        src = getfield(data, s)
        dest = getfield(data′, s)
        if s == :特征
            for t in 1:T, n in 1:N
                i = index[n, t]
                i > 0 && for f in 1:F
                    dest[f, n, t] = src[f, 1, i]
                end
            end
        elseif s == :时间戳
            for t in 1:T, n in 1:N
                dest[n, t] = dates[t]
            end
        elseif s == :代码
            for t in 1:T, n in 1:N
                dest[n, t] = codes[n]
            end
        else
            for t in 1:T, n in 1:N
                i = index[n, t]
                if i > 0
                    dest[n, t] = src[i]
                end
            end
            s == :价格 && for t in 2:T, n in 1:N
                if iszero(dest[n, t])
                    dest[n, t] = dest[n, t - 1]
                end
            end
            s == :价格 && for t in (T - 1):-1:1, n in 1:N
                if iszero(dest[n, t])
                    dest[n, t] = dest[n, t + 1]
                end
            end
        end
    end
    return data′, index
end

function pivot(src::AbstractArray{T, N}, index) where {T, N}
    dest = zeros(T, size(index))
    for n in 1:length(index)
        i = index[n]
        if i > 0
            dest[n] = src[i]
        end
    end
    return dest
end

Base.show(io::IO, ::MIME"text/plain", ::Array{<:Data}) = 
    print(io, "Array of Data")

function roll(data, rolltrn, rolltst)
    ti, tf = map(unix2date, extrema(data.时间戳))
    Δtb = Day(parsefreq(rolltrn) ÷ 3600 ÷ 24)
    Δtf = Day(parsefreq(rolltst) ÷ 3600 ÷ 24)
    inds = rollindices(ti, tf, Δtb, Δtf)
    @views ((data[:, t1], data[:, t2]) for (t1, t2) in inds)
end

function to_df(data::Data; meta_only = false, mmaparrays = false)
    fcopy = mmaparrays ? mcopy : identity
    df_meta = DataFrame()
    @showprogress "to_df..." for s in afieldnames(Data)
        s == :特征 && continue
        x = vec(getfield(data, s))
        df_meta[s] = vec(getfield(data, s))
    end
    df_meta["时间戳"] = df_meta["时间戳"].mul(1e9).astype("datetime64[ns]")
    meta_only && return df_meta
    df_fea = pd.DataFrame(pymat(data.特征), columns = featnames(data))
    return pdhcat(df_meta, df_fea)
end

function to_data(df, dst)
    try
        _to_data(df, dst)
    catch e
        rm(dst, force = true)
        throw(e)
    end
end

function _to_data(df, dst)
    isempty(df) && return
    F = length(featcols(df))
    N = df["代码"].nunique()
    T = length(df) ÷ N
    @assert N * T == length(df)
    dst = initdata(dst, F, N, T)
    h5open(dst, "r+") do fid
        @showprogress "to_data.meta..." for c in metacols(df)
            if c == "时间戳"
                x = df[c].astype("int").div(1e9).values
            elseif c == "代码"
                dfc = df[c].astype("category")
                categories = Array(dfc.cat.categories)
                codes = (dfc.cat.codes + 1).values
                x = MLString{8}[categories[i] for i in codes]
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
        特征名 = Dict(reverse(p) for p in enumerate(featcols(df)))
        write_nonarray(fid, "特征名", 特征名)
    end
    @eval GC.gc()
    return dst
end

featcols(df) = sort!(setdiff(df.columns, string.(afieldnames(Data))))

metacols(df) = sort!(setdiff(df.columns, featcols(df)))

split_metafeat(df) = df[metacols(df)], df[featcols(df)]

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