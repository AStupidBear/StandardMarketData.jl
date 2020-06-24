function discretize(h5::String; inplace = 0, name = "特征", mpiarg = ``)
    julia = joinpath(Sys.BINDIR, Base.julia_exename())
    h5disc = joinpath(@__DIR__, "h5disc.jl")
    project = joinpath(@__DIR__, "..")
    run(`mpiexec $mpiarg $julia --project=$project $h5disc --name $name --inplace $inplace $h5`)
    h5′ = replace(h5, ".h5" => "_UInt8.h5")
    inplace == 1 ? h5 : h5′
end

discretize!(a...; ka...) = discretize(a...; inplace = 1, ka...)

function discretize(x::AbstractArray; ka...)
    h5 = randstring() * ".h5"
    h5write(h5, "x", x)
    discretize!(h5; name = "x", ka...)
    x8 = h5read(h5, "x")
    bin_edges = h5read(h5, "bin_edges")
    rm(h5, force = true)
    return x8, bin_edges
end

discretize(x, edge::Vector) = map(z -> searchsortedfirst(edge, z), x)

function discretize(x, bin_edges::Matrix)
    x′ = reshape(x, size(x, 1), :)
    x8′ = zeros(UInt8, size(x′))
    for f in 1:size(x8′, 1)
        x8′[f, :] .= discretize(x′[f, :], bin_edges[:, f])
    end
    reshape(x8′, size(x))
end

function undiscretize(x8, edge::Vector)
    @inbounds map(x8) do z
        if z == 0x00
            edge[1]
        elseif z == 0xff
            edge[end]
        else
            l = edge[z]
            r = edge[z + 1]
            (l + r) / 2
        end
    end
end

function undiscretize(x8, bin_edges::Matrix)
    x8′ = reshape(x8, size(x8, 1), :)
    x′ = zeros(Float32, size(x8′))
    for f in 1:size(x′, 1)
        x′[f, :] .= undiscretize(x8′[f, :], bin_edges[:, f])
    end
    reshape(x′, size(x8))
end