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
    Threads.@threads for i ∈ 1:batchsize
        for j ∈ 1:config.ndata
            Xs[j][:,:,1,i] = wavread_process(data[1][ids[i]], config)
        end
    end
    return (Xs, map(y -> convert.(sampletype, _getobs(y, ids)), data[2:end])...)#map(Base.Fix2(_getobs, ids), data[2:end])...)
end

function _getaudioobs(data::Tuple, 
                      config::SpecConfig,
                      ids::Vector{I}) where {I<:Integer}
    batchsize = length(ids)
    Xs = [_batchinitialize(config, batchsize) for _ ∈ 1:config.ndata]
    timesec = zeros(sampletype, batchsize)
    Threads.@threads for i ∈ 1:batchsize
        x1, fs = wavread(data[1][ids[i]]; format="native")
        x = x1 |> a -> convert.(sampletype, a)
        timesec[i] = convert(sampletype, size(x, 1) / fs)
        for j ∈ 1:config.ndata
            for k ∈ 1:config.nchannels
                Xs[j][:,:,k,i] = config.augment(x[:,k]) |> a -> tospec(a, config)
            end
        end
    end
    return ((Xs, timesec), map(y -> convert.(sampletype, _getobs(y, ids)), data[2:end])...)#map(Base.Fix2(_getobs, ids), data[2:end])...)
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

function unpack_data(::TSConfig, data)
    ds = first(data)
    ds
end

function unpack_data(::SpecConfig, data)
    ds = first(data)
    first(ds)..., last(ds)
end
