using StandardMarketData
using Dates
using Test

cd(mktempdir())

F, N, T = 40, 100, 1000
h5 = initdata("test.h5", F, N, T, columns = string.("f", 1:F))
data = loaddata(h5)
savedata("tmp.h5", data)
data = reloaddata(data)

@test nfeats(data) == F
@test ncodes(data) == N
@test nticks(data) == T
@test nticksperday(data) == T
@test period(data) == 0
rescale("test.h5")
downsample(data, "1T")
@test length(featnames(data)) == F

@uncol f1, f9 = data
@test getfeat(data, r"f9") == f9
@test length(getfeats(data, r"f9")) == 1
@test nfeats(dropfeats((data, r"f9")) == F - 1
@test nfeats(keepfeats(data, r"f9")) == 1
@test nfeats(keepcats(data, r"f9")) == F
@test categories(data) == featnames(data)

datespan(data)
firstdate(data)
lastdate(data)
getlabel(data, "60T")
@test all(setcomm(data, 1).买手续费率 .== 1)
@test isempty(setpool(data, "f1"))

concat([data, data], 1)
# pivote(data)

epochsof(data)
datetimesof(data)
datesof(data)
codesof(data)

@test parsefreq("1T") == 60
@test str2date("20100101") == Date(2010, 01, 01)
@test int2unix(20100101) == datetime2unix(DateTime(2010, 01, 01))

to_data(to_df(data), "tmp.h5")
@test normalize_code("000001") == "000001.XSHE"
@test sortednunique([1,2,3,3,4]) == 4
@test isempty(roll(data, "6M", "6M"))

@test next_tradetime(DateTime("2018-01-01T8:50"), "JM") == DateTime("2018-01-01T09:00:00")
@test next_tradetime(DateTime("2018-01-01T8:50"), "000001") == DateTime("2018-01-01T09:30:00")
@test next_tradetime(DateTime("2018-01-01T12:00"), "000001") == DateTime("2018-01-01T13:00:00")
@test next_tradetime(DateTime("2018-01-01T13:30"), "JM") == DateTime("2018-01-01T13:30:00")
@test next_tradetime(DateTime("2018-01-01T13:15"), "000001") == DateTime("2018-01-01T13:15:00")
@test next_tradetime(DateTime("2018-01-01T15:15"), "000001") == DateTime("2018-01-02T09:30:00")