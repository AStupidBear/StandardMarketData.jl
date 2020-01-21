macro staticvar(init)
    var = gensym()
    __module__.eval(:(const $var = $init))
    var = esc(var)
    quote
        global $var
        $var
    end
end

function normalize_code(code)
    if occursin(".", code)
        code = replace(replace(code, "SZA" => "XSHE"), "SHA" => "XSHG")
    else
        code = lpad(code, 6, "0")
        code * ifelse(startswith(code, "6"), ".XSHG", ".XSHE")
    end
end

unix2date(t) = Date(unix2datetime(t))
unix2time(t) = Time(unix2datetime(t))

unix2str8(t) = Dates.format(unix2datetime(t), "yyyymmdd")
unix2str6(t) = Dates.format(unix2datetime(t), "yymmdd")
str2date(str) = Date(replace(str, '-' => ""), "yyyymmdd")
str2datetime(str) = DateTime(replace(str, '-' => ""), "yyyymmdd")
str2unix(str) = datetime2unix(str2datetime(str))

unix2int(t) = parse(Int, unix2str8(t))
int2unix(i) = str2unix(string(i))

isfutcode(code) = occursin(r"^\D+$", split(code, '.')[1])
iscommcode(code) = isfutcode(code) && !occursin(r"IF|IH|IC", code)

function next_tradetime(t, code)
    d, t = Date(t), Time(t)
    iscomm = iscommcode(code)
    if t < Time(9, 0) && iscomm
        t = Time(9, 0)
    elseif t < Time(9, 30) && !iscomm
        t = Time(9, 30)
    elseif Time(10, 15) <= t < Time(10, 30) && iscomm
        t = Time(10, 30)
    elseif Time(11, 30) <= t < Time(13, 0) && !iscomm
        t = Time(13, 0)
    elseif Time(13, 0) <= t < Time(13, 30) && iscomm
        t = Time(13, 30)
    elseif t >= Time(15, 0)
        t = Time(9, 30)
        d += Day(1)
    end
    t = t + d
    while true
        isholiday(t) ? t += Day(1) : return t
    end
end

function isholiday(t)
    holidays = @staticvar DateTime[]
    if isempty(holidays)
        chinese_calendar = pyimport("chinese_calendar")
        for day in keys(chinese_calendar.holidays)
            push!(holidays, day)
        end
    end
    Dates.issaturday(t) && return true
    Dates.issunday(t) && return true
    range = searchsortedfirst(holidays, Date(t))
    length(range) == 0
end

function mmap_tempname()
    isdir(".mempool") || mkdir(".mempool")
    file = joinpath(".mempool", randstring())
end

function mcopy(x::AbstractArray{T, N}) where {T, N}
    src = mmap_tempname()
    fid = open(src, "w+")
    write(fid, x)
    seekstart(fid)
    xm = Mmap.mmap(fid, Array{T, N}, size(x))
    close(fid)
    finalizer(z -> rm(src), xm)
    return xm
end

mcopy(x::Series) = Series(mcopy(x.values), index = x.index, name = x.name)

mcopy(x::DataFrame) = DataFrame(mcopy(x.values), index = x.index, columns = x.columns)

part(df::DataFrame) = nprocs() > 1 ? part(df.iloc, 1) : df

lngstconsec(s) = (!s).cumsum()[s].value_counts().max()

function reindex_columns(df, columns)
    itrs = [df[c].sort_values().unique() for c in columns]
    multidx = pd.MultiIndex.from_product(itrs, names = columns)
    df.set_index(columns, inplace = true)
    df = df.reindex(index = multidx)
    df.reset_index(inplace = true)
    return df
end

function concat_hdfs(dst, srcs; key = "df")
    store = pd.HDFStore(dst, "w")
    try
        for src in srcs
            df = pd.read_hdf(src, key)
            store.append(key, df, index = false)
        end
    finally
        store.close()
    end
    return dst
end

hasheader(src) = length(collect(eachmatch(r"\d+\.\d+", readline(src)))) <= 2

function concat_txts(dst, srcs)
    header = hasheader(srcs[1])
    open(dst, "w") do fid
        header && println(fid, readline(srcs[1]))
        @showprogress dst for src in srcs
            open(src, "r") do f
                header && readline(f)
                write(fid, read(f))
            end
        end
    end
    return dst
end

function catlag(df::DataFrame, by; maxlag = 10)
    df = pd.concat([df.groupby(by).shift(l).dropna() for l in 0:(maxlag - 1)], axis = 1)
    df.columns = [string(c, '_', l) for l in 0:(maxlag - 1) for c in df.columns]
    df.reset_index(inplace = true, drop = true)
    return df
end

function catlag(x; maxlag = 10)
    F, N, T = size(x)
    x′ = fill!(similar(x, maxlag * F, N, T), 0)
    for t in 1:T, n in 1:N, f in 1:F
        s = x[f, n, t]
        for t′ in t:min(t + maxlag - 1, T)
            x′[f + (t′ - t) * F, n, t′] = s
        end
    end
    return x′
end

function read_hdf5(src; columns = nothing, mmaparrays = true)
    df = DataFrame()
    h5open(src, "r+") do fid
        columns = something(columns, names(fid))
        @showprogress "read_hdf5..." for c in columns ∩ names(fid)
            dset = fid[c]
            if Sys.iswindows() || !mmaparrays || !ismmappable(dset)
                df[c] = read(dset)
            else
                df[c] = readmmap(dset)
            end
        end
    end
    return df
end

function to_hdf5(df, dst)
    isfile(dst) && rm(dst)
    h5open(dst, "w") do fid
        @showprogress "to_hdf5..." for c in df.columns
            fid[c] = Array(df[c])
        end
    end
    return dst
end

to_dict(x) = Dict{Symbol, Any}(s => getfield(x, s) for s in fieldnames(typeof(x)))

to_struct(::Type{T}, d) where T = T([d[s] for s in fieldnames(T)]...)

divavg!(x) = (x ./= mean(x))

function rollindices(ti, tf, Δtb, Δtf)
    [(string(t - Δtb):string(t - Day(1)),
    string(t):string(t + Δtf - Day(1)))
    for t in (ti + Δtf):Δtf:tf]
end
    
function sortednunique(f, x)
    n = 1
    yl = f(x[1])
    @inbounds for xi in x
        yi = f(xi)
        n = ifelse(yi != yl, n + 1, n)
        yl = yi
    end
    return n
end

sortednunique(x) = sortednunique(identity, x)

function sortedunique(f, x)
    n = 1
    yl = f(x[1])
    yu = [yl]
    @inbounds for xi in x
        yi = f(xi)
        if yi != yl
            n += 1
            push!(yu, yi)
        end
        yl = yi
    end
    return yu
end

sortedunique(x) = sortedunique(identity, x)

Mmap.sync!(m::AbstractArray) = isdefined(m, :parent) && Mmap.sync!(m.parent)

parsefreq(freq) = freq

function parsefreq(freq::String)
    @eval let S = 1, T = 60, H = 60T, D = 24H, M = 30D, Y = 12M
        $(Meta.parse(freq))
    end
end

splat(list) = [item for sublist in list for item in (isa(sublist, AbstractArray) ? sublist : [sublist])]

idxmap(x) = Dict(zip(x, axes(x, 1)))

⧶(x, y) = x / y
⧶(x, y::AbstractFloat) = x / (y + eps(y))
⧶(x, y::Integer) = ifelse(x == y == 0, zero(x), x / y)

function from_category(sr)
    sr = sr.astype("category")
    cats = Array(sr.cat.categories)
    codes = (sr.cat.codes + 1).values
    MLString{8}[cats[i] for i in codes]
end

function to_category(x)
    sr = pd.Series(x).astype("category")
    sr.cat.categories = sr.cat.categories.astype("str")
    return sr
end

@generated function subslice(x::AbstractArray{T, N}) where {T, N}
    inds = ntuple(i -> (:), N - 1)
    :($inds)
end

subslice(x) = ntuple(i -> (:), ndims(x) - 1)

cview(a, i) = view(a, subslice(a)..., i)

ccount(a) = (ndims(a) == 1 ? length(a) : size(a, ndims(a)))

function Base.split(x::AbstractArray, n)
    cview(x, 1:n), cview(x, (n + 1):ccount(x))
end

Base.dropdims(f, A::AbstractArray; dims) = dropdims(f(A, dims = dims), dims = dims)

function arr2rng(x)
    isempty(x) && return x
    r = UnitRange(extrema(x)...)
    r == x ? r : x
end