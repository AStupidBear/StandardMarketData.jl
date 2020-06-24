function _extract_tsfresh_feats(column, shift; ka...)
    @from tsfresh imports extract_relevant_features
    @from tsfresh.utilities.dataframe_functions imports roll_time_series
    dst = @sprintf("tsfresh/%s_%s.h5", column, shift)
    isfile(dst) && return dst
    df = pd.read_parquet("df.parquet", columns = ["代码", column])
    if df[column].dtype == "uint8"
        df[column] = df[column] / 128 - 1
    end
    y = pd.read_pickle("y.pkl")["y"]
    if length(y) == length(df)
        df = roll_time_series(df, "代码", nothing, nothing, 1, shift)
        df.drop(columns = "代码", inplace = true)
        y.index = df.groupby("id")["sort"].last().sort_values().index
        df_erf = extract_relevant_features(df, y; column_id = "id", column_sort = "sort", ka...)
    else
        df_erf = extract_relevant_features(df, y; column_id = "代码", ka...)
        df_erf.drop(columns = "代码", inplace = true)
    end
    df_erf = df_erf.astype("float32")
    df_erf.columns = ["$c-$shift" for c in df_erf.columns]
    to_hdf5(df_erf, dst)
    @eval GC.gc()
    return dst
end

function extract_tsfresh_feats(df, y = nothing; shifts = ["20T"], horizon = "20T", remove = true, ka...)
    Δt = df["时间戳"].groupby(df["代码"]).diff().median()
    shifts = @. ceil(Int, parsefreq(shifts) / Δt)
    if isnothing(y)
        horizon = ceil(Int, parsefreq(horizon) / Δt)
        y = df["涨幅"].groupby(df["代码"]).rolling(horizon).sum().groupby("代码").shift(-horizon).fillna(0)
        y = Series(y.reset_index(level = "代码", drop = true).sort_index().astype("float32"), name = "y")
    end
    df_meta, df_fea = split_metafeat(df)
    df_ts = pdhcat(df_fea, df_meta[["代码"]])
    df_ts.to_parquet("df.parquet")
    y.to_frame().to_pickle("y.pkl")
    !isdir("tsfresh") && mkdir("tsfresh")
    h5s = pmap(Iterators.product(df_fea.columns, shifts)) do (column, shift)
        _extract_tsfresh_feats(column, shift; ka...)
    end
    if length(df) != length(y)
        df = df.drop_duplicates(subset = "代码", keep = "last")
        df.reset_index(inplace = true)
        @assert length(df) == length(y)
    end
    df = pdhcat(df, read_hdf5.(h5s)...)
    remove && foreach(rm, ["df.parquet", "y.pkl", h5s...])
    return df
end

function extract_talib_feats(df, bycol = "code"; keep_columns = [])
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
        !isempty(keep_columns) && c ∉ keep_columns && return
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

function TradingFCParameters(tscal = 1)
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
        # "c3" => [Dict("lag" => tscal * lag) for lag in [1, 5]],
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