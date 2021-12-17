export gettargets

"""
Initialize a batch 4-D array of all zeros with size of (`winsize`, 1, `config.nchannels`, `batchsize`).

# Arguments
config: Config instance
batchsize: batch size

# Returns
- a 4-D array of all zeros
"""
function _batchinitialize(config::TSConfig, batchsize::Int)
    zeros(sampletype, config.winsize, 1, config.nchannels, batchsize)
end
function _batchinitialize(config::SpecConfig, batchsize::Int)
    zeros(sampletype, config.newdims..., config.nchannels, batchsize)
end

"""
Get minibatches of audio files with indices `ids` in the form specified by `config` and the 
corresponding targets.

# Arguments
- data : Tuple contains WAV paths and the corresponding targets
- config: Config instance
- ids: indices of audio files

# Returns
- minibatches in a tuple of audio data, time lengths, sampling rates 
- vectors containing the corresponding targets.   
"""
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

"""
Transform a vector of time-series data to a scaled spectrogram specified by `config`.

# Arguments
- x : time-series audio data
- config: SpecConfig instance

# Returns
- a scaled spectrogram specified by `config`
"""
function tospec(x::AbstractVector, config::SpecConfig)
    spec = stft(x, config.winsize, config.noverlap; window=config.window) |>
           a -> config.scaled(a)::Matrix{sampletype}
    imresize(spec, config.newdims...)
end

"""
Unpack audio data into a tuple of audio data, time lengths, sampling rates 
by selecting the first augmented view.

# Arguments
- data : acoustic data with multiple augmented views
- withtimesec : if `true`, includes time lengths (in seconds) of audio files in seconds 
- withsamplingrate : if `true`, includes sampling rates of audio files in samples

# Returns
- audio data in the form of first augmented view of the acoustic data and the corresponding time 
lengths and sampling rates   
"""
function unpack_data(data, withtimesec::Bool, withsamplingrate::Bool)
    ds = first(data)
    if withtimesec && withsamplingrate
        first(first(ds)), ds[2], ds[3]
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

# Arguments
- d : AudioLoader instance

# Returns
- a vector of vectors containing all the targets in `d`
"""
function gettargets(d::AudioLoader)
    targets = d.data[2:end]
    n = length(targets)
    d.shuffle && (targets = Tuple([target[d.indices] for target ∈ targets]))
    n == 1 ? only(targets) : targets
end