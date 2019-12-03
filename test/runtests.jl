next_tradetime(DateTime("2018-01-01T8:50"), "JM")
next_tradetime(DateTime("2018-01-01T8:50"), "000001")
next_tradetime(DateTime("2018-01-01T12:00"), "000001")
next_tradetime(DateTime("2018-01-01T13:15"), "JM")
next_tradetime(DateTime("2018-01-01T13:15"), "000001")
next_tradetime(DateTime("2018-01-01T15:15"), "000001")

using HDF5, HDF5Utils

h5open("/tmp/test.h5", "w") do fid
    d_zeros(fid, "a", MLString{8}, 10, 10)
    fid["a"][:, :] .= MLString{8}("CC")
    # fid["a"] = [ MLString{8}("") for i in 1:10]
end

h5open("/tmp/test.h5", "r") do fid
    read(fid["a"])
end