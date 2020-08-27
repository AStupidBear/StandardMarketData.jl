macro staticvar(init)
    var = gensym()
    __module__.eval(:(const $var = $init))
    var = esc(var)
    quote
        global $var
        $var
    end
end

macro staticdef(ex)
    @capture(ex, name_::T_ = val_) || error("invalid @staticvar")
    ref = Ref{__module__.eval(T)}()
    set = Ref(false)
    :($(esc(name)) = if $set[]
        $ref[]
    else
        $ref[] = $(esc(ex))
        $set[] = true
        $ref[]
    end)
end

unix2date(t) = Date(unix2datetime(t))
unix2time(t) = Time(unix2datetime(t))
unix2hour(x) = x % (24 * 3600) / 3600

unix2str8(t) = Dates.format(unix2datetime(t), "yyyymmdd")
unix2str6(t) = Dates.format(unix2datetime(t), "yymmdd")
str2date(str) = Date(replace(str, '-' => ""), "yyyymmdd")
str2datetime(str) = DateTime(replace(str, '-' => ""), "yyyymmdd")
str2unix(str) = datetime2unix(str2datetime(str))

unix2int(t) = parse(Int, unix2str8(t))
int2unix(i) = str2unix(string(i))

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
        @showprogress 10 dst for src in srcs
            open(src, "r") do f
                header && readline(f)
                write(fid, read(f))
            end
        end
    end
    return dst
end

function concat_lagged(df::DataFrame, by; maxlag = 10)
    df = pd.concat([df.groupby(by).shift(l).dropna() for l in 0:(maxlag - 1)], axis = 1)
    df.columns = [string(c, '_', l) for l in 0:(maxlag - 1) for c in df.columns]
    df.reset_index(inplace = true, drop = true)
    return df
end

function concat_lagged(x; maxlag = 10)
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
        @showprogress 10 "read_hdf5..." for c in columns ∩ names(fid)
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
        @showprogress 10 "to_hdf5..." for c in df.columns
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

nunique(x) = length(Set(x))
    
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

if PandasLite.version() >= v"0.25"
    function to_category(x::AbstractArray{<:MLString})
        sr = pd.Series(x).astype("category")
        sr.cat.categories = sr.cat.categories.str.decode("utf-8", "ignore")
        return sr
    end
end

Base.dropdims(f, A::AbstractArray; dims) = dropdims(f(A, dims = dims), dims = dims)

function arr2rng(x)
    isempty(x) && return x
    r = UnitRange(extrema(x)...)
    r == x ? r : x
end

isna(x) = iszero(x) | isnan(x)