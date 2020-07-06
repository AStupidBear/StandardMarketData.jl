#!/usr/env/bin julia
using ProgressMeter
using HDF5
using HDF5Utils
using MPI
using ArgParse
using Statistics
using SortingAlgorithms
using Parameters
using Mmap

function part(x::AbstractArray{T, N}, dim = -1) where {T, N}
    dim = dim > 0 ? dim : N + dim + 1
    dsize = size(x, dim)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    wsize = MPI.Comm_size(MPI.COMM_WORLD)
    @assert wsize <= dsize
    chunk = ceil(Int, dsize / wsize)
    is = (rank * chunk + 1):min(dsize, (rank + 1) * chunk)  
    view(x, ntuple(x -> x == dim ? is : (:), N)...)
end

function discretize!(x8, x)
    bin_edges = [Float32[] for f in 1:size(x, 1)]
    prob = range(0, 1, length = 256 + 1)[2:(end - 1)]
    prob′ = range(0, 1, length = 2560 + 1)[2:(end - 1)]
    prog = Progress(size(x, 1), desc = "BinEdges: ")
    for f in 1:size(x, 1)
        xf = filter(z -> !isnan(z) & !isinf(z), x[f, :])
        sort!(xf, alg = RadixSort)
        v = quantile(xf, prob′, sorted = true)
        v = MPI.Allgather(v, MPI.COMM_WORLD)
        bin_edges[f] = quantile(v, prob)
        if MPI.Comm_rank(MPI.COMM_WORLD) == 0
            next!(prog)
        end
    end
    prog = Progress(size(x, 2), desc = "Discretize: ")
    for n in 1:size(x, 2)
        @inbounds for f in 1:size(x, 1)
            edge = bin_edges[f]
            x8[f, n] = searchsortedfirst(edge, x[f, n]) - 1
        end
        if MPI.Comm_rank(MPI.COMM_WORLD) == 0
            next!(prog)
        end
    end
    MPI.Barrier(MPI.COMM_WORLD)
    return bin_edges
end

MPI.Init()

setting = ArgParseSettings()
@add_arg_table setting begin
    "--name"
        arg_type = String
        required = true
    "--inplace"
        arg_type = Int
        default = 0   
    "h5"
        arg_type = String
        required = true
end
@unpack name, inplace, h5 = parse_args(ARGS, setting)
h5′ = replace(h5, ".h5" => "_UInt8.h5")

x = h5readmmap(h5, name, mode = "r")
if eltype(x) == UInt8
    MPI.Finalize()
    exit()
end

if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    h5open(h5′, "w", "alignment", (0, 8)) do fid′
        h5open(h5, "r") do fid
            for dset in names(fid)
                dset == name && continue
                o_copy(fid[dset], fid′, dset)
            end
        end
        d_zeros(fid′, name, UInt8, size(x))
    end
end
MPI.Barrier(MPI.COMM_WORLD)

x8 = h5readmmap(h5′, name, mode = "r+")
x = reshape(x, size(x, 1), :)
x8 = reshape(x8, size(x8, 1), :)

bin_edges = discretize!(part(x8), part(x))

if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    Mmap.sync!(x8)
    h5open(h5′, "r+") do fid′
        fid′["bin_edges"] = hcat(bin_edges...)
    end
    inplace == 1 && mv(h5′, h5, force = true)
end

MPI.Finalize()