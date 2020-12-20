creshape(x) = reshape(x, :, size(x)[end])
rreshape(x) = reshape(x, size(x, 1), :)

csize(a) = (ndims(a) == 1 ? size(a) : size(a)[1:end-1])
csize(a, n) = tuple(csize(a)..., n) # size if you had n columns
rsize(a) = (ndims(a) == 1 ? size(a) : size(a)[2:end])
rsize(a, n) = tuple(n, rsize(a)...) # size if you had n columns

ccount(a) = (ndims(a) == 1 ? length(a) : size(a, ndims(a)))
rcount(a) = (ndims(a) == 1 ? length(a) : size(a, 1))

@generated function subslice(x::AbstractArray{T, N}) where {T, N}
    inds = ntuple(i -> (:), N - 1)
    :($inds)
end
subslice(x) = ntuple(i -> (:), ndims(x) - 1)

cview(a, i) = view(a, subslice(a)..., i)
rview(a, i) = view(a, i, subslice(a)...)
cget(a, i) = getindex(a, subslice(a)..., i)
rget(a, i) = getindex(a, i, subslice(a)...)
cset!(a, x, i) = setindex!(a, x, subslice(a)..., i)
rset!(a, x, i) = setindex!(a, x, i, subslice(a)...)

unsqueeze(x; dims) = reshape(x, ntuple(d -> d == dims ? 1 : d < dims ? size(x, d) : size(x, d - 1), Val(ndims(x) + 1)))

function concatenate(xs::Vector{<:AbstractArray{T, N}}; dims = -1) where {T, N}
    shape = ntuple(Val(N)) do d
        d != dims ? size(xs[1], d) :
        sum(size.(xs, dims))
    end
    xc, i = zeros(T, shape), 0
    for x in xs
        is = ntuple(Val(N)) do d
            d != dims ? (1:shape[d]) :
            (i + 1):(i + size(x, dims))
        end
        copyto!(view(xc, is...), x)
        i += size(x, dims)
    end
    return xc
end

cconcatenate(xs) = concatenate(xs, dims = ndims(xs[1]))
rconcatenate(xs) = concatenate(xs, dims = 1)

stack(xs; dims) = concatenate(unsqueeze.(xs, dims = dims), dims = dims)
unstack(xs; dims) = [selectdim(xs, dims, i) for i in 1:size(xs, dims)]

cstack(xs) = stack(xs, ndims(first(xs)) + 1)
rstack(xs) = stack(xs, 1)

indbatch(x, b, offset = 0) = (C = ccount(x); min(i + offset, C):min(i + offset + b -1, C) for i in 1:b:C)
minibatch(x, batchsize) = [cview(x, ind) for ind in indbatch(x, batchsize)]

function Base.split(x::AbstractArray, n::Integer)
    cview(x, 1:n), cview(x, (n + 1):ccount(x))
end

function Base.split(x::AbstractArray, is::AbstractArray{<:Integer})
    [cview(x, i:i′) for (i, i′) in zip([0; is[1:end - 1]] .+ 1, is)]
end

colvec(x) = reshape(x, length(x), 1)
rowvec(x) = reshape(x, 1, length(x))

splat(list) = [item for sublist in list for item in (isa(sublist, AbstractArray) ? sublist : [sublist])]
