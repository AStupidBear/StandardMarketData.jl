using StandardMarketData
using HDF5Utils
using PandasLite
using Statistics
using Dates
using Test

cd(mktempdir())

F, N, T = 2, 5, 100

特征名 = idxmap(string.("f", 1:F))
特征 = randn(Float32, F, N, T)
涨幅 = dropdims(mean(特征, dims = 1), dims = 1)
买手续费率 = 卖手续费率 = fill(1f-4, N, T)
涨停 = 跌停 = zeros(Float32, N, T)
代码 = MLString{8}[string(2t <= T ? n : N + n) for n in 1:N, t in 1:T]
ti, Δt = DateTime(2019, 1, 1), Hour(1)
时间戳 = range(ti, step = Δt, length = T ÷ 2)
时间戳 = datetime2unix.(repeat(reshape(时间戳, 1, :), N, 2))
价格 = cumsum(涨幅, dims = 2)
交易池 = ones(Float32, N, T)
data = Data(特征名, 特征, 涨幅, 买手续费率, 卖手续费率, 涨停, 跌停, 代码, 时间戳, 价格, 交易池)

savedata("test.h5", data)
@test data == reloaddata(data)
@test nticks(downsample(data, "5H")) == T ÷ 5
@test size(pivot(data)[1]) == (2N, T ÷ 2)
@test isapprox(SMD.getlabel(data, 5)[:, 10], sum(涨幅[:, 11:15], dims = 2))

@test nfeats(data) == F
@test ncodes(data) == N
@test nticks(data) == T
@test period(data) == Second(Δt).value
@test length(featnames(data)) == F

@test ncodes(data["1"]) == 1
data["20190101":"20190103"]
split(data, "20190110")

@uncol f1, f2 = data
@test isapprox(getfeat(data, r"f1$"), 特征[1, :, :])
@test getfeats(data, r"f1$") == [特征[1, :, :]]
@test nfeats(dropfeats(data, r"f1$")) == F - 1
@test nfeats(keepfeats(data, r"f1$")) == 1
@test nfeats(keepcats(data, r"f1$")) == F
@test getcats(data) == featnames(data)

df = to_df(data)
@test df["f1"].values == vec(f1)
to_data(df, "tmp.h5")
@test all(isone, setcomm(data, 1).买手续费率)
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

@test normalize_code("000001") == "000001.XSHE"
@test nunique([1, 2, 2, 3]) == 3
@test sortednunique([1, 2, 3, 3, 4]) == 4
@test isempty(rolldata(data, "6M", "6M"))

@test next_tradetime(DateTime("2018-01-01T8:50"), "JM") == DateTime("2018-01-01T09:00:00")
@test next_tradetime(DateTime("2018-01-01T8:50"), "000001") == DateTime("2018-01-01T09:30:00")
@test next_tradetime(DateTime("2018-01-01T12:00"), "000001") == DateTime("2018-01-01T13:00:00")
@test next_tradetime(DateTime("2018-01-01T13:30"), "JM") == DateTime("2018-01-01T13:30:00")
@test next_tradetime(DateTime("2018-01-01T13:15"), "000001") == DateTime("2018-01-01T13:15:00")
@test next_tradetime(DateTime("2018-01-01T15:15"), "000001") == DateTime("2018-01-02T09:30:00")

if !isnothing(Sys.which("mpiexec"))
    x = data.特征
    x8, bin_edges = discretize(x, host = "localhost")
    x′ = undiscretize(x8, bin_edges)
    @test mean(abs, x .- x′) < 0.01
end

extract_tsfresh_feats(to_df(data), shifts = ["10H"], horizon = "3H")

df = DataFrame("code" => vec(代码), "close" => vec(价格))
df["high"] = df["low"] = df["open"] = df["close"]
df["volume"] = rand(N * T)
extract_talib_feats(df, "code")