using StandardMarketData
using HDF5Utils
using PandasLite
using Statistics
using Dates
using Random
using Test

cd(mktempdir())

Random.seed!(1234)
F, N, T = 2, 5, 100

特征名 = idxmap(string.("f", 1:F))
特征 = randn(Float32, F, N, T)
涨幅 = dropdims(mean(特征, dims = 1), dims = 1)
特征[:, 1:1, 1:3] .= NaN
特征[:, 2:2, 1:3] .= Inf
ti, Δt = DateTime(2019, 1, 1), Hour(1)
时间戳 = range(ti, step = Δt, length = T ÷ 2)
时间戳 = datetime2unix.(repeat(reshape(时间戳, 1, :), N, 2))
代码 = MLString{8}[string(2t <= T ? n : N + n) for n in 1:N, t in 1:T]
最新价 = 买1价 = 卖1价 = cumsum(涨幅, dims = 2)
手续费率 = fill(1f-4, N, T)
涨停 = 跌停 = zeros(Float32, N, T)
交易池 = ones(Float32, N, T)
data = Data(特征名, 特征, 涨幅, 时间戳, 代码, 最新价, 买1价, 卖1价, 手续费率, 涨停, 跌停, 交易池)

savedata("test.h5", data)
@test data == reloaddata(data)
@test nticks(downsample(data, "5H")) == T ÷ 5
Sys.iswindows() || @test size(pivot(data)[1]) == (2N, T ÷ 2)
@test isapprox(pct_change(data, 5)[:, 10], sum(涨幅[:, 11:15], dims = 2))

@test nfeats(data) == F
@test ncodes(data) == N
@test nticks(data) == T
@test period(data) == Second(Δt).value
@test length(featnames(data)) == F

@test ncodes(data["1"]) == 1
data["20190101":"20190103"]
split(data, "20190110")

@uncol f1, f2 = data
@test isapprox(getfeat(data, r"f1$"), 特征[1, :, :], nans = true)
@test isequal(getfeats(data, r"f1$"), [特征[1, :, :]])
@test nfeats(dropfeats(data, r"f1$")) == F - 1
@test nfeats(keepfeats(data, r"f1$")) == 1
@test nfeats(keepcats(data, r"f1$")) == F
@test getcats(data) == featnames(data)

df = to_df(data)
@test isequal(df["f1"].values, vec(f1))
to_data(df, "tmp.h5")
@test all(isone, setcomm(data, 1).手续费率)
@test isempty(setpool(data, "f1"))

@test firstdate(data) == minimum(unix2date, 时间戳)
@test lastdate(data) == maximum(unix2date, 时间戳)
@test datespan(data) == (firstdate(data), lastdate(data))

@test ncodes(concat([data, data], 1)) == 2N
@test nticks(concat([data, data], 2)) == 2T

@test epochsof(data) == unique(时间戳)
@test datetimesof(data) == unix2datetime.(epochsof(data))
@test datesof(data) == unix2date.(epochsof(data))
@test Set(codesof(data)) == Set(unique(代码))

@test parsefreq("1T") == 60
@test str2date("20100101") == Date(2010, 01, 01)
@test int2unix(20100101) == datetime2unix(DateTime(2010, 01, 01))

@test nunique([1, 2, 2, 3]) == 3
@test sortednunique([1, 2, 3, 3, 4]) == 4
@test isempty(rolldata(data, "6M", "6M"))

if PandasLite.version() >= v"0.25"
    extract_tsfresh_feats(to_df(data), shifts = ["10H"], horizon = "3H")
end

if !Sys.iswindows()
    df = DataFrame("code" => vec(代码), "close" => vec(最新价))
    df["high"] = df["low"] = df["open"] = df["close"]
    df["volume"] = rand(N * T)
    extract_talib_feats(df, "code")
end

if !isnothing(Sys.which("mpiexec"))
    x = data.特征
    x8, bin_edges = discretize(x, mpiarg = `-n 1`)
    x′ = undiscretize(x8, bin_edges)
    @test mean(x .- x′) do z
        abs(!isnan(z) * !isinf(z) * z)
    end < 0.02
end

# foo(x) = unsqueeze(x; dims = 2)
# x = rand(10, 10)
# @code_warntype foo(x)

# xs = [rand(10, 10) for i in 1:2]
# concatenate(xs, dims = 1)
# foo(xs) = concatenate(xs, dims = 1)
# foo(xs); @code_warntype foo(xs)