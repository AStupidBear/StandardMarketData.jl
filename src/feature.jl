function cluster(x, n; method = "mbkmns", n_jobs = 1)
    if method == "kmns"
        @from dask_ml.cluster imports KMeans
        model = KMeans(n_clusters = n, random_state = 0, n_jobs = n_jobs)
    elseif method == "mbkmns"
        @from dask_ml.cluster imports PartialMiniBatchKMeans
        model = MiniBatchKMeans(n_clusters = n, random_state = 0)
    elseif method == "birch"
        @from sklearn.cluster imports Birch
        model = Birch(n_clusters = n)
    elseif method == "agglm"
        @from sklearn.cluster imports AgglomerativeClustering
        model = AgglomerativeClustering(n_clusters = n)
    end
    y = model.fit_predict(pymat(x))
    y = Float32.(y) .+ 1
    reshape(y, 1, size(x)[2:end]...)
end

function discretize(x, n; encode = "ordinal", strategy = "quantile")
    @from sklearn.preprocessing imports KBinsDiscretizer
    model = KBinsDiscretizer(n_bins = n, encode = encode, strategy = strategy)
    y = Float32.(model.fit_transform(pymat(x)))
    if encode == "ordinal"
        subtoind = LinearIndices(ntuple(z -> n, size(x, 1)))
        y = Float32[getindex(subtoind, Int.(y[t, :] .+ 1)...) for t in 1:size(y, 1)]
        reshape(y, 1, size(x)[2:end]...)
    elseif encode == "onehot-dense"
        reshape(permutedims(y), size(y, 2), size(x)[2:end]...)
    end
end

function reducedims(x, n; method = "tsne", nbr = 32, n_jobs = 1)
    if method == "tsne"
        if sizeof(x) > 1024^3 || n != 2
            @from MulticoreTSNE imports MulticoreTSNE
            model = MulticoreTSNE(n_components = n, random_state = 0, n_jobs = n_jobs)
        else
            @from tsnecuda imports TSNE
            model = TSNE(num_neighbors = nbr)
        end
    elseif method == "pca"
        @from dask_ml.decomposition imports PCA
        model = PCA(n_components = n)
    elseif method == "kpca"
        @from sklearn.decomposition imports KernelPCA
        model = KernelPCA(n_components = n, kernel = "rbf", random_state = 0, n_jobs = n_jobs)
    elseif method == "isomap"
        @from sklearn.manifold imports Isomap
        model = Isomap(n_components = n, n_neighbors = nbr, n_jobs = n_jobs)
    elseif method == "mds"
        @from sklearn.manifold imports MDS
        model = MDS(n_components = n, random_state = 0, n_jobs = n_jobs)
    elseif method == "se"
        @from sklearn.manifold imports SpectralEmbedding
        model = SpectralEmbedding(n_components = n, n_neighbors = nbr, random_state = 0, n_jobs = n_jobs)
    elseif method[1:3] == "lle"
        @from sklearn.manifold imprts LocallyLinearEmbedding
        # lle_standard, lle_ltsa, lle_hessian, lle_modified
        model = LocallyLinearEmbedding(n_components = 2, n_neighbors = nbr, method = method[5:end], random_state = 0, n_jobs = n_jobs)
    end
    y = pycall(model.fit_transform, PyArray, pymat(x))
    reshape(y, :, size(x)[2:end]...)
end

isbin(x) = !any(z -> z != 0 && z != 1, x)

function winsorize!(df::DataFrame, mode = "standard")
    if Threads.nthreads() <= 1
        @showprogress "winsorize..." for s in df
            df[s].nunique() > 2 && winsorize!(values(df[s]), mode)
        end
    else
        cs = String[c for c in df.columns]
        xs = [values(df[c]) for c in cs]
        Threads.@threads for n in 1:length(xs)
            c, x = cs[n], xs[n]
            threadprint("winsorizing $c")
            !isbin(x) && winsorize!(x, mode)
        end
    end
end

function droplowvar!(df, threshold = 0.0)
    dropcols = [s for s in df if df[s].std() < threshold]
    @info "dropping columns" dropcols = dropcols
    df.drop(inplace = true, columns = dropcols)
    return df
end

function clean!(df)
    df.dropna(inplace = true, axis = 1, how = "all")
    df.fillna(inplace = true, method = "ffill")
    df.fillna(inplace = true, method = "bfill")
    winsorize!(df, "standard")
    droplowvar!(df, 1e-4)
end

function trans_1arg(df)
    isempty(df) && return df
    @from sklearn.preprocessing imports PowerTransformer
    @from dask_ml.preprocessing imports QuantileTransformer
    pt, qt = PowerTransformer(), QuantileTransformer()
    df_1arg = DataFrame()
    @showprogress "trans_1arg..." for c in df
        df_1arg["power($c)"] = mcopy(pt.fit_transform(df[c]))
        df_1arg["quantile($c)"] = mcopy(qt.fit_transform(df[c]))
        df_1arg["log1p($c)"] = mcopy(log1p.(Array(df[c])))
        df_1arg["sqrt($c)"] = mcopy(sqrt.(Array(df[c])))
    end
    return df_1arg
end

function trans_2arg(df)
    isempty(df) && return df
    df_2arg = DataFrame()
    @showprogress "trans_2arg..." for c1 in df.columns, c2 in df.columns
        c1 >= c2 && continue
        df_2arg["-($c1, $c2)"] = mcopy(df[c1] - df[c2])
        df_2arg["/($c1, $c2)"] = mcopy(df[c1] / df[c2])
        df_2arg["/($c1, $c2)"] = mcopy(df[c2] / df[c1])
        df_2arg["rdiff($c1, $c2)"] = mcopy((df[c1] - df[c2]) / (df[c1] + df[c2]))
    end
    df_2arg.fillna(0, inplace = true)
    return df_2arg
end

function trans_poly(df)
    isempty(df) && return df
    df_poly = DataFrame()
    @showprogress "trans_poly..." for c1 in df.columns, c2 in df.columns
        c1 > c2 && continue
        df_poly["$c1 * $c2"] = mcopy(df[c1] * df[c2])
    end
    return df_poly
end

function trans_kbin(df, nbins = 6)
    isempty(df) && return df
    @from sklearn.preprocessing imports KBinsDiscretizer
    kbins = KBinsDiscretizer(n_bins = nbins, strategy = "quantile")
    df_kbin = DataFrame()
    @showprogress "trans_kbin..." for c in df.columns
        xbin = kbins.fit_transform(df[c])
        for n in 1:nbins
            df_kbin["$c[$n]"] = mcopy(xt.getcol(n).toarray())
        end
    end
    return df_kbin
end

function _trans_ts_(slc, col, win, tscal)
    dst = @sprintf("tmp/%s_%s_%s.h5", slc, col, win)
    isfile(dst) && parseenv("RESUME", false) && return dst
    df = read_hdf5("pmap.h5", columns = ["id", col])
    nid = df["id"].nunique()
    isbin = df[col].nunique() <= 2
    dfs = Any[]
    if parseenv("TS_ROLL", false)
        ti, tf = slc[1], slc[end]
        ti != 1 && (ti -= win * nid)
        @assert ti >= 1
        df_slc = df.iloc[ti:tf, :]
        aggfuns = isbin ? ["mean"] : ["max", "mean", "min", "std"]
        # ["kurt", "max", "mean", "median", "min", "skew", "std", "var"]
        for (n, dfn) in df_slc.groupby("id")
            dfn.drop(inplace = true, columns = "id")
            df_roll = dfn.rolling(win).agg(aggfuns)
            df_roll.columns = [join(c, '_') for c in df_roll.columns]
            for c in dfn.columns
                df_roll["Δ$c"] = dfn[c].diff(win)
            end
            push!(dfs, df_roll)
        end
        df_concat = pd.concat(dfs, axis = 0)
        df_concat.sort_index(inplace = true)
        ti != 1 && (df_concat = df_concat.tail(-win * nid))
    else
        @from tsfresh imports extract_features
        fc_params = TradingFCParameters(tscal, isbin)
        for t in IterTools.takenth(slc, nid)
            df_t = iloc(df)[(t - win * nid):t, :]
            df_ef = extract_features(df_t, column_id = "id",
                        n_jobs = 0, disable_progressbar = true,
                        default_fc_parameters = fc_params)
            df_ef = DataFrame(df_ef.astype("float"))
            push!(dfs, df_ef)
        end
        ti = findfirst(!isempty, dfs)
        dfs[1:(ti - 1)] .= Ref(dfs[ti])
        df_concat = pdvcat(dfs...)
    end
    @show slc, col, win
    df_concat.columns = ["$win-$c" for c in df_concat.columns]
    to_hdf5(df_concat, dst)
    @eval GC.gc()
    return dst
end

function trans_ts(df, args...)
    if isfile("trans_ts.h5") && parseenv("RESUME", false)
        return read_hdf5("trans_ts.h5")
    end
    to_hdf5(df, "pmap.h5")
    cols = setdiff(df.columns, ["id"])
    slcs = indbatch(1:length(df), 100000nunique(df["id"]))
    args = [slcs, cols, args...]
    srcs = pmap(x -> _trans_ts_(x...), Iterators.product(args...))
    df = pd.read_hdf5(h5concat("trans_ts.h5", srcs))
    rm("pmap.h5"); foreach(rm, srcs)
    return df
end

function TradingFCParameters(tscal, isbin)
    if isbin
        Dict(
            "last_location_of_maximum" => nothing,
            "mean" => nothing,
            "mean_abs_change" => nothing,
        )
    else
        Dict(
            "abs_energy" => nothing,
            "count_above_mean" => nothing,
            "first_location_of_maximum" => nothing,
            "first_location_of_minimum" => nothing,
            "kurtosis" => nothing,
            "last_location_of_maximum" => nothing,
            "last_location_of_minimum" => nothing,
            "longest_strike_above_mean" => nothing,
            "longest_strike_below_mean" => nothing,
            "maximum" => nothing,
            "mean" => nothing,
            "mean_abs_change" => nothing,
            "mean_change" => nothing,
            "mean_second_derivative_central" => nothing,
            "median" => nothing,
            "minimum" => nothing,
            # "sample_entropy" => nothing,
            "skewness" => nothing,
            "standard_deviation" => nothing,
            "variance" => nothing,
            # "time_reversal_asymmetry_statistic" => [Dict("lag" => tscal * lag) for lag in [1, 5]],
            # "c3" => [Dict("lag" => tscl * lag) for lag in [1, 5]],
            # "symmetry_looking" => [Dict("r" => r * 0.05) for r in [1, 5, 10]],
            # "large_standard_deviation" => [Dict("r" => r * 0.05) for r in [1, 5, 10]],
            # "quantile" => [Dict("q" => q) for q in [0.25, 0.75]],
            # "autocorrelation" => [Dict("lag" => tscal * lag) for lag in [1, 5, 10]],
            # "agg_autocorrelation" => [Dict("f_agg" => s, "maxlag" => 40 * tscal) for s in ["mean", "median", "var"]],
            # "partial_autocorrelation" => [Dict("lag" => tscal * lag) for lag in [1, 5, 10]],
            # !"number_peaks" => [Dict("n" => tscal * n) for n in [1, 5, 10]],
            "binned_entropy" => [Dict("max_bins" => 10)],
            # !"fft_aggregated" => [Dict("aggtype" => s) for s in ["centroid", "variance", "skew", "kurtosis"]],
            # "max_langevin_fixed_point" => [Dict("m" => 3, "r" => 30)],
            # !"linear_trend" => [Dict("attr" => "pvalue"), Dict("attr" => "rvalue"), Dict("attr" => "slope"), Dict("attr" => "stderr")],
            # "agg_linear_trend" => [Dict("attr" => attr, "chunk_len" => tscal * i, "f_agg" => f)
            #                      for attr in ["rvalue", "slope", "stderr"]
            #                      for i in [5, 10, 50]
            #                      for f in ["max", "min", "mean", "var"]],
            # "augmented_dickey_fuller" => [Dict("attr" => "teststat"), Dict("attr" => "pvalue"), Dict("attr" => "usedlag")],
            # "ratio_beyond_r_sigma" => [Dict("r" => x) for x in [1, 5]],
            # "ar_coefficient" => [Dict("coeff" => coeff, "k" => k) for coeff in 0:4 for k in [10]],
        )
    end
end

function trans_cluster(df, nclus = 50)
    @from dask_ml.cluster imports PartialMiniBatchKMeans
    @from dask_ml.preprocessing imports OneHotEncoder
    kmns = MiniBatchKMeans(n_clusters = nclus)
    enc = OneHotEncoder(sparse = false)
    labels = enc.fit_transform(reshape(kmns.fit_predict(df), :, 1))
    inertia = kmns.transform(df) ./ sqrt(size(df, 2))
    df_labels = DataFrame(mcopy(labels), columns = ["cluster[$i]" for i in 1:nclus])
    df_inertia = DataFrame(mcopy(inertia), columns = ["inertia[$i]" for i in 1:nclus])
    pdhcat(df_labels, df_inertia)
end

function trans_kernel(df, ncmpts = 100)
    @from sklearn.kernel_approximation imports RBFSampler
    @from sklearn.kernel_approximation imports Nystroem
    rbf = RBFSampler(n_components = ncmpts)
    nys = Nystroem(n_components = ncmpts)
    df_rbf = DataFrame(mcopy(rbf.fit_transform(df)), columns = ["rbf[$n]" for n in 1:ncmpts])
    df_nys = DataFrame(mcopy(nys.fit_transform(df)), columns = ["nys[$n]" for n in 1:ncmpts])
    df_kern = pdhcat(df_rbf, df_nys)
    return df_kern
end

function trans_dim(df, ncmpts = 100)
    @from dask_ml.decomposition imports PCA
    pca = PCA(n_components = 100)
    DataFrame(mcopy(pca.fit_transform(df)), columns = ["pca[$n]" for n in 1:ncmpts])
end

function afe(df; wins = [1])
    df_meta, df_fea = split_metafeat(df)
    df_id = DataFrame(Dict("id" => df_meta["股票代码"]))
    winsorize!(df_fea, "standard")
    df_ts = pdhcat(df_id, df_fea)
    df_ts = trans_ts(df_ts, wins, minimum(wins))
    df_fea = clean!(pdhcat(df_fea, df_ts))
    df_afe = pdhcat(df_meta, df_fea)
end

# function afe(df_meta, df_fea; tscal = 5, nl = false, disc = false, wins = [tscal])
#     reset_timer!()
#     @timeit "winsorize df_fea" winsorize!(df_fea, "minmax")
#     @timeit "trans_1arg" df_1arg = nl ? trans_1arg(df_fea) : DataFrame()
#     @timeit "trans_2arg" df_2arg = nl ? trans_2arg(df_fea) : DataFrame()
#     @timeit "trans_poly" df_poly = nl ? trans_poly(df_fea) : DataFrame()
#     @timeit "trans_kbin" df_bin = disc ? trans_kbin(df_fea) : DataFrame()
#     @timeit "trans_polykbin" df_bin = disc ? droplowvar!(trans_poly(df_bin)) : DataFrame()
#     df_num = pdhcat(df_fea, df_1arg, df_2arg, df_poly)
#     @timeit "winsorize df_num" winsorize!(df_num, "standard")
#     df_id = Series(df_meta["股票代码"], name = "id")
#     df_num["id"] = df_bin["id"] = df_id
#     print_timer()
#     @timeit "trans_ts df_num" df_num_ts = trans_ts(df_num, wins, tscal, false)
#     @timeit "trans_ts df_bin" df_bin_ts = disc ? trans_ts(df_bin, 60, tscal, true) : DataFrame()
#     df_num.drop(inplace = true, columns = "id")
#     df_bin.drop(inplace = true, columns = "id")
#     df_ts = pdhcat(df_num_ts, df_bin_ts)
#     dropna(df_ts, inplace = true, axis = 1, how = "all")
#     fillna(df_ts, inplace = true, method = "ffill")
#     fillna(df_ts, inplace = true, method = "bfill")
#     @timeit "winsorize df_num_ts" winsorize!(df_num_ts, "standard")
#     df_afe = pdhcat(df_num, df_bin, df_ts)
#     droplowvar!(df_afe, 1e-4)
#     # df_clus = trans_cluster(df_afe)
#     # df_kern = trans_kernel(df_afe)
#     # df_afe = pdhcat(df_afe, df_clus, df_kern)
#     # df_afe = trans_dim(df_afe)
#     df_afe["dummy"] = 1
#     @timeit "to_hdf5 df_afe" to_hdf5(df_afe, "df_afe.h5")
#     print_timer()
#     println(describe(df_afe))
#     return df_afe
# end

function talib(df, bycol = "code"; keepcols = [])
    ohlc_cols = ["open", "high", "low", "close", "volume"]
    df_ohlc = df[ohlc_cols].astype("double")
    df_talib = df[df.columns.drop(ohlc_cols)]
    @from talib imports get_function_groups
    normclose = [get_function_groups()["Overlap Studies"];
                get_function_groups()["Price Transform"];
                ["APO", "MACD", "MACDEXT", "MACDFIX", "MINUS_DM", "MOM", "PLUS_DM"];
                ["TRANGE", "ATR", "LINEARREG", "LINEARREG_INTERCEPT", "STDDEV", "TSF"]]
    groupfuncs = [(g, f) for (g, fs) in get_function_groups() for f in fs]
    dfs = @showprogress pmap(groupfuncs) do (g, f)
        g = split(g, ' ')[1]
        g == "Math" && return
        f == "MAVP" && return
        f == "VAR" && return
        c = "talib:" * g * "_" * f
        !isempty(keepcols) && c ∉ keepcols && return
        println(c)
        py"""
        import gc
        import pandas as pd
        from talib.abstract import Function
        def apptalib(x):
            df = Function($f)(x)
            if len(df.shape) == 1:
                df = df.to_frame()
            return df
        df = $df_ohlc.groupby($df_talib[$bycol]).apply(apptalib)
        if $f in $$normclose:
            for c in df.columns:
                df[c] = df[c] / $df_ohlc["close"]
        df = df.astype('float32')
        df.columns = [$c if c == 0 else $c + '_' + c  for c in df]
        gc.collect()
"""
        return py"df"
    end
    return pdhcat(df_talib, filter(!isnothing, dfs)...)
end