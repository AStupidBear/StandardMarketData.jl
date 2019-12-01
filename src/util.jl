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

function next_tradetime(code, t)
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
        for day in keys(chinese_calendar[:holidays])
            push!(holidays, day)
        end
    end
    Dates.issaturday(t) && return true
    Dates.issunday(t) && return true
    range = searchsortedfirst(holidays, Date(t))
    length(range) == 0
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

part(df::DataFrame) = nprocs() > 1 ? part(df.iloc, dim = 1) : df

function hdfconcat(dst, srcs; key = "df")
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

lngstconsec(s) = (!s).cumsum()[s].value_counts().max()

function reindex_columns(df, cols)
    itrs = [df[c].sort_values().unique() for c in cols]
    multidx = pd.MultiIndex.from_product(itrs, names = cols)
    df.set_index(cols, inplace = true)
    df = df.reindex(index = multidx)
    df.reset_index(inplace = true)
    return df
end

hasheader(src) = length(collect(eachmatch(r"\d+\.\d+", readline(src)))) <= 2

function txtconcat(dst, srcs)
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

function catlag(df::DataFrame, gcol, maxlag = 10)
    dfs = [df.groupby(gcol).shift(l) for l in 0:(maxlag - 1)]
    cols = [string(c, '_', n) for n in 0:(maxlag - 1) for c in dfs[1].columns]
    df = pd.concat(dfs, axis = 1).dropna()
    df.reset_index(inplace = true, drop = true)
    df.columns = cols
    return df
end

function catlag(x, maxlag = 10)
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

downsample(df::PandasWrapped, freq::String) =
    downsample(to_hdf5(df, "resample_pre.h5"), freq)

function downsample(src::String, freq::String)
    fmap = @static Sys.iswindows() ? map : pmap
    @printf("downsampling %s...\n", src)
    srcs = fmap(h5open(names, src)) do c
        df = pd.read_hdf5(src, columns = [c, "日期"])
        Δt = df["日期"].iloc[10] - df["日期"].iloc[9]
        @printf("%s, freq %.1gS => %s\n", c, Δt, freq)
        t = pd.date_range(start = "19700101", periods = length(df), freq = string(Δt, "S"))
        df.set_index(t, inplace = true)
        orient = c == "涨幅" ? "left" : "right"
        r = df[[c]].resample(freq, label = orient, closed = orient)
        aggf = c == "涨幅" ? "sum" : occursin(r"手续费|日期", c) ? "last" : "mean"
        Sys.iswindows() && (c = randstring())
        r.agg(aggf).to_hdf5("tmp/$c.h5")
    end
    df = read_hdf5(h5concat("resample.h5", srcs))
    df["涨幅"].fillna(0, inplace = true)
    @eval GC.gc(true)
    return df
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