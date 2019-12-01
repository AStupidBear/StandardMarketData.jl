mutable struct Data{A <: AbstractMatrix{Float32},
                    B <: AbstractMatrix{Float32},
                    C <: AbstractMatrix{Float32},
                    D <: AbstractMatrix{Float64},
                    E <: AbstractArray{Float32, 3},
                    F <: AbstractMatrix{Float32},
                    G <: AbstractMatrix{Float32},
                    H <: AbstractMatrix{Float32},
                    I <: AbstractMatrix{Float32}}
    src::String
    # 股票专用
    特征列::Dict{String, Int}   # 特征每列名称
    股票代码::A                 # 股票代码列表, 总股票数x天数
    股票池::B                   # 股票池, 总股票数x天数, 1表示在股票池中
    指数::C                     # (BH,上证50,沪深300,中证500)X天数
    # 通用
    日期::D                     # 总股票数x天数
    特征::E                     # 特征, 特征x总股票数x天数
    涨幅::F                     # 归一化日涨跌幅, 百分数, 总股票数x天数
    买手续费率::G               # 手续费率, 百分数, 总股票数x天数
    卖手续费率::G               # 手续费率, 百分数, 总股票数x天数
    涨停::H                    # 是否跌停, 涨停为1, 总股票数x天数
    跌停::H                     # 是否跌停, 跌停为1, 总股票数x天数
    价格::I                     # 净值, 总股票数x天数
end

afieldnames(t) = Symbol[fieldname(t, n) for n in 1:fieldcount(t) if fieldtype(t, n) <: AbstractArray]

colnames(data) = first.(sort(collect(data.特征列), by = last))

transidx(s, is) = s == :特征 ? (:, is...) :
                s == :指数 ? (:, is[2:end]...) : is

function trange(data, rng)
    ti = str2unix(rng.start)
    tf = str2unix(rng.stop)
    ts = data.日期[1, :]
    ti .<= ts .<= tf
end

Base.lastindex(data::Data, n) = size(data)[n]

Base.getindex(data::Data, is...) =
    Data("", data.特征列, [getindex(getfield(data, s), transidx(s, is)...) for s in afieldnames(Data)]...)
Base.getindex(data::Data, ir, ic::StringRange) = getindex(data, ir, trange(data, ic))
Base.getindex(data::Data, code::String) = getindex(data, :, vec(data.股票代码) .== str2flt(code))

Base.view(data::Data, is...) =
    Data("", data.特征列, [view(getfield(data, s), transidx(s, is)...) for s in afieldnames(Data)]...)
Base.view(data::Data, ir, ic::StringRange) = view(data, ir, trange(data, ic))
Base.maybeview(data::Data, is...) = view(data, is...)

Base.reshape(data::Data, is...) =
    Data("", data.特征列, [s == :指数 ? similar(getfield(data, s), 4, is[end]) :
     reshape(getfield(data, s), transidx(s, is)...) for s in afieldnames(Data)]...)
Base.vec(data::Data) = reshape(data, 1, length(data))

Base.copy(data::Data) =
    Data([s == :src ? "" : getfield(data, s) for s in fieldnames(Data)]...)

function nfeas(data)
    F = size(data.特征, 1)
    F > 1 && return F
    xmax = -Inf32
    for x in data.特征
        if modf(x)[1] != 0
            return F
        else
            xmax = max(xmax, x)
        end
    end
    return Int(xmax)
end
nstocks(data) = size(data.特征, 2)
nticks(data) = size(data.特征, 3)
Base.size(data::Data) = size(data.涨幅)
Base.length(data::Data) = prod(size(data))

ind2date(data, t) = unix2date(data.日期[1, t])

ndays(data) = sortednunique(unix2date, @view(data.日期[1, :]))

nticksperday(data) = nticks(data) ÷ ndays(data)

period(data) = nticks(data) > 1 ? median(diff(@view(data.日期[1, :]))) : 1

isstocks(data) = any(!isone, data.指数)

function pat2srcs(pat)
    startswith(pat, '/') && return [pat]
    pwds, h5s = glob(pat), glob(pat, DROOT)
    !isempty(pwds) ? pwds : !isempty(h5s) ? h5s :
    error("cannot find data matching $pat")
end

pat2src(pat) = first(pat2srcs(pat))

const nrmcoefmap = Dict{String, NTuple{4, Float32}}()

function normalize_index(data)
    if length(data.指数) == 0 || size(data.指数, 1) > 4
        dict = todict(data)
        dict[:指数] = ones(Float32, 4, nticks(data))
        tostruct(Data, dict)
    else
        data
    end
end

function _loaddata(src; mode = "r", ti = nothing, tf = nothing, ka...)
    if endswith(src, ".h5")
        data = h5load(src, Data; mode = mode, ka...)
        if "归一系数" ∈ h5open(names, data.src)
            coeff = h5read(data.src, "归一系数")
        end
    elseif endswith(src, ".bson")
        data = bsload(src, Data; ka...)
    end
    data = normalize_index(data)
    @isdefined(coeff) && for (c, f) in data.特征列
        nrmcoefmap[c] = tuple(coeff[:, f]...)
    end
    isnothing(ti) && isnothing(tf) && return data
    ti = something(ti, "20000101")
    tf = something(tf, "20501231")
    @view data[:, ti:tf]
end

function loaddata(srcs::AbstractArray; dim = -1, ka...)
    datas = @showprogress map(srcs) do src
        _loaddata(src; ka...)
    end
    length(srcs) == 1 && return datas[1]
    concat(filter(!isempty, datas), dim)
end

loaddata(pat; ka...) = loaddata(pat2srcs(pat); ka...)

function savedata(dst, data)
    if endswith(dst, ".bson")
        bssave(dst, data[:, :])
    elseif endswith(dst, ".h5")
        if all(isone, data.指数)
            h5save(dst, data, excludes = [:指数])
        else
            h5save(dst, data)
        end
    end
end

function reloaddata(data)
    dst = @sprintf("data-%s.h5", randstring())
    savedata(dst, data, force = true)
    loaddata(dst, mode = "r+")
end

function initdata(dst, F, N, T; columns = [])
    isfile(dst) && rm(dst)
    h5open(dst, "w") do fid
        g_create(fid, "nonarray")
        @showprogress "initdata..." for s in afieldnames(Data)
            if s == :指数
                d_zeros(fid, string(s), Float32, 4, T)
            elseif s == :日期
                d_zeros(fid, string(s), Float64, N, T)
            elseif s == :特征
                d_zeros(fid, string(s), Float32, F, N, T)
            else
                d_zeros(fid, string(s), Float32, N, T)
            end
        end
        fid["股票代码"][:, :] .= 1:N
        fid["股票池"][:, :] = 1
        fid["指数"][:, :] = 1
        if !isempty(columns)
            特征列 = Dict(reverse(p) for p in enumerate(columns))
            write_nonarray(fid, "特征列", 特征列)
        end
    end
    return dst
end

function Base.show(io::IO, data::Data)
    compact = get(io, :compact, false)
    @printf(io, "文件路径: %s\n", data.src)
    @printf(io, "特征数: %d\t", nfeas(data))
    @printf(io, "品种数: %d\t", nstocks(data))
    @printf(io, "TK数: %d\n", nticks(data))
    @printf(io, "股票池比例: %.2g\t", mean(data.股票池))
    @printf(io, "涨停比例: %.2g\t", mean(data.涨停))
    @printf(io, "跌停比例: %.2g\n", mean(data.跌停))
    ts = Dates.format.(unix2date.(data.日期[[1;end]]), "yymmdd")
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
    for (f, i) in data.特征列
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

Mmap.sync!(m::AbstractArray) = hasfield(m, :parent) && Mmap.sync!(m.parent)

function rescale!(srcs; fillnan = true, filter_cols = [])
    isa(srcs, AbstractArray) || (srcs = [srcs])
    datas = loaddata.(srcs, mode = "r+")
    F, N, T = size(datas[1].特征)
    filter_cols = filter_cols == true ? keys(datas[1].特征列) :
                filter_cols == false ? [] : filter_cols
    block_feas = [get(datas[1].特征列, c, -1) for c in filter_cols]
    used_feas = setdiff(1:F, block_feas)
    nrmcoef = zeros(Float32, 4, F)
    @showprogress "computing nrmcoef..." for f in 1:F
        if f ∈ used_feas
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
        else
            nrmcoef[:, f] = [-Inf32, Inf32, 0f0, 1f0]
        end
    end
    for data in datas
        name = @sprintf("normalizing %s...", basename(data.src))
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
        h5write(data.src, "归一系数", nrmcoef)
    end
end

function rescale(h5)
    h5′ = replace(h5, ".h5" => "_NRM.h5")
    run(`cp $h5 $h5′`)
    rescale!(h5′)
end

parsefreq(freq) = freq

function parsefreq(freq::String)
    @eval let S = 1, T = 60, H = 60T, D = 24H, M = 30D, Y = 12M
        $(Meta.parse(freq))
    end
end

downsample(data::Data, freq::String; ka...) =
    downsample(data, round(Int, parsefreq(freq) / period(data)); ka...)

function downsample(data::Data, freq::Int; phase = 1, ka...)
    freq <= 1 && return data[:, :]
    downsample(data, phase:freq:nticks(data); ka...)
end

function downsample(data::Data, ts::AbstractArray{Int}; average = false)
    data′ = data[:, ts]
    comp = any(!isone, data.指数)
    F, N, T = size(data.特征)
    fill!(data′.涨幅, comp ? 1 : 0)
    @showprogress for t′ in 1:(length(ts) - 1)
        @inbounds for t in ts[t′]:(ts[t′ + 1] - 1), n in 1:N
            Δ = data.涨幅[n, t]
            Δ′ = data′.涨幅[n, t′]
            data′.涨幅[n, t′] = ifelse(comp, Δ′ * (1 + Δ), Δ′ + Δ)
        end
    end
    if length(ts) == 1
        copyto!(data′.涨幅, sum(data.涨幅, dims = 2))
    end
    data′.涨幅 .-= comp ? 1 : 0
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

function prune(data; seperate = false)
    data = seperate ? vec(data) : data
    keep = zeros(Bool, nticks(data))
    for t in 1:nticks(data)
        keep[t] = any(iszero, @view(data.涨停[:, t])) || any(iszero, @view(data.跌停[:, t]))
    end
    tp, data′ = 0, data[:, keep]
    for t in 1:nticks(data)
        if keep[t]
            tp += 1
        elseif tp > 0
            for n in 1:nstocks(data)
                data′.涨幅[n, tp] += data.涨幅[n, t]
            end
        end
    end
    clamp!(data′.涨幅, -2f-1, 2f-1)
    if nstocks(data′) == 1
        @assert maximum(data′.涨停 .+ data′.跌停) < 2
    end
    return data′
end

function col(data::Data, c; denorm = true, errors = "raise")
    if c in keys(data.特征列)
        f = data.特征列[c]
        x = @view data.特征[f, :, :]
        if haskey(nrmcoefmap, c) && denorm
            ~, ~, μ, σ = nrmcoefmap[c]
            x = x .* σ .+ μ
        end
        return x
    elseif Symbol(c) in fieldnames(typeof(data))
        getfield(data, Symbol(c))
    elseif errors == "ignore"
        zeros(Float32, size(data))
    else
        error("$c not in data or data.特征列")
    end
end

function col(data::Data, r::Regex; ka...)
    cs = cols(data, r; ka...)
    !isempty(cs) &&  return cs[1]
    col(data, string(r); ka...)
end

cols(data::Data, cs; ka...) = [col(data, c; ka...) for c in cs]
cols(data::Data, r::Regex; ka...) = cols(data, filter(c -> occursin(r, c), colnames(data)); ka...)

function dropcols(data, cols::AbstractArray{Int})
    isempty(cols) && return data
    @unpack 特征, 特征列 = data
    fs = setdiff(1:nfeas(data), cols)
    特征 = view(特征, fs, :, :)
    cols = first.(sort(collect(特征列), by = last)[fs])
    特征列 = Dict(zip(cols, axes(cols, 1)))
    Data("", 特征列, [s == :特征 ? 特征 : getfield(data, s) for s in afieldnames(Data)]...)
end
dropcols(data, cols::Array{<:AbstractString}) = dropcols(data, [get(data.特征列, c, -1) for c in cols])
dropcols(data, r::Regex) = dropcols(data, filter(c -> occursin(r, c), colnames(data)))
keepcols(data, cols::AbstractArray{Int}) = dropcols(data, setdiff(1:nfeas(data), cols))
keepcols(data, cols::Array{<:AbstractString}) = dropcols(data, setdiff(colnames(data), cols))
keepcols(data, r::Regex) = keepcols(data, filter(c -> occursin(r, c), colnames(data)))

categories(data) = filter(!isempty, unique(String.(first.(split.(colnames(data), ':')))))
keepcats(data, cats) = keepcols(data, filter(c -> any(cat -> startswith(c, cat) || !occursin(':', c), splat(cats)), colnames(data)))
dropcats(data, cats) = keepcats(data, setdiff(categories(data), splat(cats)))
dropcats(data, r::Regex) = dropcats(data, filter(c -> occursin(r, c), categories(data)))
keepcats(data, r::Regex) = keepcats(data, filter(c -> occursin(r, c), categories(data)))

macro uncol(ex)
    cols, rhs = ex.args
    d, cols = gensym(), isa(cols, Symbol) ? [cols] : cols.args
    kd = [:($c = $col($d, $(string(c)))) for c in cols]
    Expr(:block, :($d = $rhs), kd..., d) |> esc
end

function Base.split(data::Data, tstr::String)
    epoch = str2unix(tstr)
    t = findfirst(x -> epoch <= x, data.日期[1, :])
    @views data[:, 1:(t - 1)], data[:, t:end]
end

function fixcomm(data)
    p, r = data.价格, zero(data.涨幅)
    p̄ = mean(p, dims = 2)
    N, T = size(data)
    for t in 1:(T - 1), n in 1:N
        r[n, t] = (p[n, t + 1] - p[n, t]) / p̄[n]
    end
    dict = todict(data)
    dict[:涨幅] = r
    return tostruct(Data, dict)
end

datespan(data::Data) = unix2date.(data.日期[1, [1, end]])
firstdate(data::Data) = unix2date(data.日期[1, 1])
lastdate(data::Data) = unix2date(data.日期[1, end])

getlabel(data::Data, h::String) = getlabel(data, ceil(Int, parsefreq(h) / period(data)))

function getlabel(data::Data, h::Int)
    h > nticks(data) && return data.涨幅
    @unpack 涨幅, 股票代码 = data
    N, T = size(data)
    l = zeros(Float32, N, T)
    r = zeros(Float32, N)
    for t in T:-1:(T - h + 1), n in 1:N
        r[n] += 涨幅[n, t]
        l[n, t] = r[n] / (T - t + 1)
    end
    @showprogress "getlabel..." for t in (T - h):-1:1, n in 1:N
        if 股票代码[n, t] != 股票代码[n, t + 1]
            r[n] = 涨幅[n, t]
        else
            r[n] += 涨幅[n, t] - 涨幅[n, t + h]
        end
        l[n, t] = r[n] / h
    end
    return l
end

function setcomm(data, c)
    dict = todict(data)
    for s in [:买手续费率,  :卖手续费率]
        dict[s] = fill(Float32(c), size(data))
    end
    tostruct(Data, dict)
end

function setpool(data, pool)
    ns = findall(vec(any(isone, pool, dims = 2)))
    data′, pool′ = data[ns, :], pool[ns, :]
    data′.股票池 = isone.(pool′)
    return data′
end

setpool(data, c::Union{String, Regex}) = setpool(data, col(data, c))

function Base.repeat(data::Data, n)
    n == 1 && return data
    fvs = []
    for s in fieldnames(Data)
        x = getfield(data, s)
        if isa(x, AbstractMatrix)
            nb = s == :指数 ? 1 : n
            bsizes = (fill(size(x, 1), nb), [size(x, 2)])
            x = _BlockArray(repeat([x], nb, 1), bsizes...)
        end
        push!(fvs, x)
    end
    return Data(fvs...)
end

epochsof(data::Data) = unique(data.日期)
timesof(data::Data) = unix2datetime.(unique(data.日期))
datesof(data::Data) = unix2date.(unique(data.日期))
codesof(data::Data) = flt2str.(unique(data.股票代码))

normalmask(data::Data) = @. iszero(data.涨停) | iszero(data.跌停)

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
        elseif s == :src
            x = ""
        else
            x = x1
        end
        push!(fvs, x)
    end
    return Data(fvs...)
end

rebatch(data::Data, batchsize) = normalize_index(Data("", data.特征列, [rebatch(getfield(data, s), batchsize) for s in afieldnames(Data)]...))

Base.isempty(data::Data) = length(data) == 0

pivot(datas::AbstractArray{<:Data}) = concat(map(vec, datas), -1)

function pivot(data::Data; dst = "pivot.h5")
    data = vec(data)
    @unpack 股票代码, 日期 = data
    codes = sort(unique(股票代码))
    dates = sort(unique(日期))
    midx = pd.MultiIndex.from_product((dates, codes));
    df = DataFrame("code" => vec(股票代码), "date" => vec(日期));
    df = df.reset_index().set_index(["date", "code"]).reindex(midx)
    df["index"] = df["index"].fillna(-1).astype("int") + 1
    F, N, T = nfeas(data), length(codes), length(dates)
    index = reshape(df["index"].values, N, T)
    initdata(dst, F, N, T, columns = colnames(data))
    data′ = loaddata(dst, mode = "r+");
    fill!(data′.涨停, 1)
    fill!(data′.跌停, 1)
    fill!(data′.股票池, 0)
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
        elseif s == :日期
            for t in 1:T, n in 1:N
                dest[n, t] = dates[t]
            end
        elseif s == :股票代码
            for t in 1:T, n in 1:N
                dest[n, t] = codes[n]
            end
        elseif s != :指数
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

function rollindices(ti, tf, Δtb, Δtf)
    [(string(t - Δtb):string(t - Day(1)),
    string(t):string(t + Δtf - Day(1)))
    for t in (ti + Δtf):Δtf:tf]
end

function roll(data, rolltrn, rolltst)
    ti, tf = map(unix2date, extrema(data.日期))
    Δtb = Day(parsefreq(rolltrn) ÷ 3600 ÷ 24)
    Δtf = Day(parsefreq(rolltst) ÷ 3600 ÷ 24)
    inds = rollindices(ti, tf, Δtb, Δtf)
    @views ((data[:, t1], data[:, t2]) for (t1, t2) in inds)
end

divavg!(x) = (x ./= mean(x))

function reweight_pnl(data)
    date = lastdayofmonth.(unix2datetime.(data.日期))
    # pnl = get(记忆收益率, hash(data.日期)) do
    #     abs.(getlabel(data, "4H"))
    # end
    pnl = abs.(getlabel(data, "4H"))
    df = DataFrame("date" => vec(date), "pnl" => vec(pnl))
    df.set_index("date", inplace = true)
    df["pnl_avg"] = df["pnl"].groupby("date").mean()
    w = (1 / df["pnl_avg"].abs()).values
    clamp!(w, quantile(w, (0.05, 0.95))...)
    reshape(divavg!(w), size(data))
end

function reweight_time(data, scale = "2Y")
    ts = data.日期
    λ = 1 / parsefreq(scale)
    ti, tf = extrema(ts)
    divavg!(@. exp(λ * (ts - tf)))
end

reweight(data) = reweight_time(data) .* reweight_pnl(data)

function to_df(data::Data; meta_only = false, mmaparrays = false)
    fcopy = mmaparrays ? mcopy : identity
    @unpack 特征列, 特征, 指数 = data
    df = DataFrame()
    @showprogress "to_df.meta..." for s in afieldnames(Data)
        s in [:特征, :指数] && continue
        x = vec(getfield(data, s))
        if s == :日期
            df[s] = pd.to_datetime(1e9 .* x)
        elseif s == :股票代码
            df[s] = flt2str.(x)
        else
            df[s] = copy(x)
        end
    end
    meta_only || @showprogress "to_df.fea..." for (c, f) in 特征列
        df[c] = fcopy(vec(特征[f, :, :]))
    end
    if isstocks(data)
        pools = ["BH", "SZ50", "HS300", "ZZ500"]
        for (n, c) in zip(1:size(指数, 1), pools)
            x = repeat(指数[n, :], inner = nstocks(data))
            df[c * "指数"] = fcopy(x)
        end
    end
    return df
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
    F = length(feacols(df))
    N = df["股票代码"].nunique()
    T = length(df) ÷ N
    @assert N * T == length(df)
    dst = initdata(dst, F, N, T)
    h5open(dst, "r+") do fid
        @showprogress "to_data.meta..." for c in setdiff(metacols(df), indcols(df))
            if c == "日期"
                x = df[c].astype("int").div(1e9).values
            elseif c == "股票代码"
                x = df[c].astype("category").cat.rename_categories(str2flt).astype("float32").values
            else
                x = df[c].values
            end
            fid[c][:, :] = reshape(x, N, :)
        end
        fid["涨停"][:, end] = 0
        fid["跌停"][:, end] = 0
        for (n, c) in enumerate(["BH", "SZ50", "HS300", "ZZ500"])
            if c * "指数" in df.columns
                fid["指数"][n, :] = Array(df[c * "指数"])[1:N:end]
            end
        end
        !any(c -> occursin("指数", c), df.columns) && o_delete(fid, "指数")
        Δt = 1024^3 ÷ (4 * F * N)
        rows = LinearIndices((1:N, 1:T))
        @showprogress "to_data.fea..." for t in 1:Δt:T
            ti, tf = t, min(t + Δt - 1, T)
            ri, rf = rows[1, ti], rows[end, tf]
            slc = df.loc[(ri - 1):(rf - 2), feacols(df)]
            slc = slc.astype("float32").T.values
            x = reshape(slc, F, N, :)
            fid["特征"][:, :, ti:tf] = x
        end
        特征列 = Dict(reverse(p) for p in enumerate(feacols(df)))
        write_nonarray(fid, "特征列", 特征列)
    end
    @eval GC.gc()
    return dst
end

feacols(df) = sort!(setdiff(df.columns, string.(afieldnames(Data)), indcols(df)))
metacols(df) = sort!(setdiff(df.columns, feacols(df)))

indcols(df) = sort!(String[c for c in df.columns if occursin("指数", c)])
split_metafea(df) = df[metacols(df)], df[feacols(df)]