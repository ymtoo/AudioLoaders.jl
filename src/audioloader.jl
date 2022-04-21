export AudioLoader

struct AudioLoader{D,C,R<:AbstractRNG}
    data::D
    config::C
    batchsize::Int
    nobs::Int
    partial::Bool
    imax::Int
    indices::Vector{Int}
    shuffle::Bool
    rng::R
end

"""
DataLoader for audio files and the respective assignments. Each mini-batch contains 
`batchsize` observations. 

`data` represents audio files in a vector of paths or arrays. 
`config` contains either window size `winsize`, sample type `sampletype`, data augmentation 
`augment`, number of augmented data `ndata` for time-series mini-batch or with additional 
parameters including number of overlap `noverlap`, window `win`, scaled function `scaled` 
for spectrogram mini-batch. The implementation is based on 
https://github.com/FluxML/Flux.jl/blob/0a215462ad8e0ba795205c9e94864403207d63fa/src/data/dataloader.jl
"""
function AudioLoader(data,
                     config; 
                     batchsize=1,
                     shuffle=false,
                     partial=true,
                     rng=GLOBAL_RNG)
    n = numobs(data)
    if n < batchsize
        @warn "Number of observations less than batchsize, decreasing the batchsize to $n"
        batchsize = n
    end
    imax = partial ? n : n - batchsize + 1
    AudioLoader(data,
                config, 
                batchsize, 
                n, 
                partial, 
                imax, 
                [1:n;], 
                shuffle, 
                rng)
end

function Base.iterate(d::AudioLoader, i=0)
    i >= d.imax && return nothing
    if d.shuffle && i == 0
        shuffle!(d.rng, d.indices)
    end
    nexti = min(i + d.batchsize, d.nobs)
    ids = d.indices[i+1:nexti]
    batch = _getaudioobs(d.data, d.config, ids)
    return (batch, nexti)
end

function Base.getindex(d::AudioLoader, i::Integer)
    iterate(d, (i - 1) * d.batchsize) |> first
end

function Base.length(d::AudioLoader)
    isempty(d) && return 0
    n = d.nobs / d.batchsize
    d.partial ? ceil(Int, n) : floor(Int, n)
end

Base.firstindex(d::AudioLoader) = 1
Base.lastindex(d::AudioLoader) = length(d)
Base.isempty(d::AudioLoader) = any(isempty.(d.data))
function Base.getindex(d::AudioLoader, indices::UnitRange)
    n = d.nobs
    i = (first(indices) - 1) * d.batchsize + 1 
    j = min((last(indices) - 1) * d.batchsize + 1 + d.batchsize, d.nobs)
    #j = (last(indices) - 1) * d.batchsize
    d.partial && (n > j) && (n - j < d.batchsize) && (j = n)
    ((i > n) || (j > n)) && (return throw(BoundsError(d, i:j)))
    newdata = []
    for d ∈ d.data
        nd = ndims(d)
        colons = [Colon() for _ ∈ 1:nd-1]
        push!(newdata, d[colons...,i:j])
    end
    AudioLoader(Tuple(newdata), 
                d.config; 
                batchsize = d.batchsize, 
                shuffle = d.shuffle, 
                partial = d.partial, 
                rng = d.rng)
end
