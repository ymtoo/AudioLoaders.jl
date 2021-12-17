export gettargets

function _batchinitialize(config::TSConfig, batchsize::Int)
    zeros(sampletype, config.winsize, 1, config.nchannels, batchsize)
end

function _batchinitialize(config::SpecConfig, batchsize::Int)
    zeros(sampletype, config.newdims..., config.nchannels, batchsize)
end

function _getaudioobs(data::Tuple, 
                      config::TSConfig,
                      ids::Vector{I}) where {I<:Integer}
    batchsize = length(ids)
    Xs = [_batchinitialize(config, batchsize) for _ ∈ 1:config.ndata]
    timesec = zeros(sampletype, batchsize)
    samplingrates = zeros(sampletype, batchsize)
    Threads.@threads for i ∈ 1:batchsize
        xsize = first(wavread(data[1][ids[i]]; format="size"))
        _, fs = wavread(data[1][ids[i]]; subrange=1)
        timesec[i] = convert(sampletype, xsize / fs)
        samplingrates[i] = convert(sampletype, fs)
        for j ∈ 1:config.ndata
            Xs[j][:,:,1,i] = wavread_process(data[1][ids[i]], config)
        end
    end
    return ((Xs, timesec, samplingrates), map(y -> _getobs(y, ids), data[2:end])...)#map(Base.Fix2(_getobs, ids), data[2:end])...)
end

function _getaudioobs(data::Tuple, 
                      config::SpecConfig,
                      ids::Vector{I}) where {I<:Integer}
    batchsize = length(ids)
    Xs = [_batchinitialize(config, batchsize) for _ ∈ 1:config.ndata]
    timesec = zeros(sampletype, batchsize)
    samplingrates = zeros(sampletype, batchsize)
    Threads.@threads for i ∈ 1:batchsize
        x1, fs = wavread(data[1][ids[i]]; format="native")
        x = convert.(sampletype, x1) #|> a -> convert.(sampletype, a)
        timesec[i] = convert(sampletype, size(x, 1) / fs)
        samplingrates[i] = convert(sampletype, fs)
        for j ∈ 1:config.ndata, k ∈ 1:config.nchannels
            Xs[j][:,:,k,i] = config.augment(x[:,k]) |> a -> tospec(a, config)
        end
    end
    return ((Xs, timesec, samplingrates), map(y -> _getobs(y, ids), data[2:end])...)#map(Base.Fix2(_getobs, ids), data[2:end])...)#
end

function wavread_process(wavpath::AbstractString, config::TSConfig)
    wavlen = first(wavread(wavpath; format="size"))::Int
    if wavlen ≤ config.winsize
        x = wavread(wavpath; format="native") |> 
            first |> 
            a -> convert.(sampletype, a)
        wavlen == config.winsize && (return x[:,1:config.nchannels])
        npad = config.winsize - wavlen
        nleftpad = config.randsegment ? rand(1:npad) : npad ÷ 2
        nrightpad = npad - nleftpad
        return [zeros(sampletype, nleftpad, config.nchannels);
                x[:,1:config.nchannels];
                zeros(sampletype, nrightpad, config.nchannels)]
    else # wavlen > config.winsize
        nextra = wavlen - config.winsize
        startind = config.randsegment ? rand(1:nextra) : ceil(Int, nextra/2)
        x = wavread(wavpath, subrange=(1+startind):(startind+config.winsize); format="native") |>
            first |>
            a -> convert.(sampletype, a)
        return x[:,1:config.nchannels]
    end
end

function tospec(x::AbstractVector, config::SpecConfig)
    spec = stft(x, config.winsize, config.noverlap; window=config.window) |>
           a -> config.scaled(a)::Matrix{sampletype}
    imresize(spec, config.newdims...)
end

function unpack_data(data, withtimesec, withsamplingrate)
    ds = first(data)
    if withtimesec && withsamplingrate
        first(first(ds)), ds[2:end]...
    elseif withtimesec
        first(first(ds)), ds[2]
    elseif withsamplingrate
        first(first(ds)), ds[3]
    else
        (first(first(ds)),)
    end
end

"""
Returns targets of an AudioLoader.
"""
function gettargets(d::AudioLoader)
    (_, ys...) = d[1]
    ts = eltype.(ys)
    targets = Vector{Real}[t[] for t ∈ ts]
    for (_, ys...) ∈ d
        for (y, target) ∈ zip(ys, targets)
            append!(target, y)
        end
    end
    targets
end