export freqmaxpool_padsegment, gettargets

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

function freqmaxpool(x::AbstractMatrix{T}, newnfreq::Int) where {T}
    nfreq = size(x,1)
    if nfreq == newnfreq
        x 
    else
        mp = nfreq ÷ newnfreq # to the nearest integer 
        mod(nfreq, newnfreq) != 0 && ((nfreq - 1) ÷ newnfreq != mp) && throw(ArgumentError("Invalid newdims for frequency"))
        MaxPool((mp,1))(reshape(x, size(x)..., 1, 1))[:,:,1,1]
    end 
end

function freqmaxpool_padsegment(x::AbstractMatrix{T}, newdims::Tuple; type::Symbol=:center) where {T}
    nfreq, ntime = size(x)
    xr = if nfreq == newdims[1] 
            x 
         else
            mp = nfreq ÷ newdims[1] # to the nearest integer 
            mod(nfreq, newdims[1]) != 0 && ((nfreq - 1) ÷ newdims[1] != mp) && throw(ArgumentError("Invalid newdims for frequency"))
            MaxPool((mp,1))(reshape(x, size(x)..., 1, 1))[:,:,1,1]
         end#imresize(x, (newdims[1], ntime))
    if newdims[2] > ntime # pad
        m = newdims[2] - ntime
        y = fill(minimum(xr), newdims) #zeros(T, newdims...)
        if type == :center
            start = 1 + m ÷ 2
            y[:,start:start+ntime-1] = xr
        elseif type == :random
            start = rand(1:m)
            y[:,start:start+ntime-1] = xr
        else
            throw(ArgumentError("Invalid padsegment type"))
        end
        y
    elseif newdims[2] < ntime # segment
        m = ntime - newdims[2]
        if type == :center
            start = 1 + m ÷ 2
        elseif type == :random
            start = rand(1:m)
        else
            throw(ArgumentError("Invalid padsegment type"))
        end
        xr[:,start:start+newdims[2]-1]
        #xr[:,1:newdims[2]]
    else
        xr
    end
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
    #Xs = [Array{sampletype}[] for _ ∈ 1:config.ndata]
    Xs = [_batchinitialize(config, batchsize) for _ ∈ 1:config.ndata]
    timesec = zeros(sampletype, batchsize)
    samplingrates = zeros(sampletype, batchsize)
    Threads.@threads for i ∈ 1:batchsize
        #xsize = first(wavread(data[1][ids[i]]; format="size"))
        x1, fs = wavread(data[1][ids[i]]; format="native")
        x = convert.(sampletype, x1)
        timesec[i] = convert(sampletype, size(x, 1) / fs)
        samplingrates[i] = convert(sampletype, fs)
        for j ∈ 1:config.ndata, k ∈ 1:config.nchannels
            Xs[j][:,:,k,i] = config.preprocess(signal(x[:,k], fs)) |>
                             s -> rand_padsegment(s, config) |>
                             s -> config.augment(s) 
            #push!(Xs[j], config.augment(x, fs)) #wavread_process(data[1][ids[i]], config)
        end
    end
    return ((Xs, timesec, samplingrates), map(y -> getobs(y, ids), data[2:end])...)#map(Base.Fix2(_getobs, ids), data[2:end])...)
end
function _getaudioobs(data::Tuple, 
                      config::SpecConfig,
                      ids::Vector{I}) where {I<:Integer}
    batchsize = length(ids)
    Xs = [_batchinitialize(config, batchsize) for _ ∈ 1:config.ndata]
    timesec = zeros(sampletype, batchsize)
    samplingrates = zeros(sampletype, batchsize)
    nstride = config.winsize - config.noverlap
    n = nstride * config.newdims[2] + nstride
    Threads.@threads for i ∈ 1:batchsize
        x1, fs = wavread(data[1][ids[i]]; format="native")
        x = convert.(sampletype, x1) 
        timesec[i] = convert(sampletype, size(x, 1) / fs)
        samplingrates[i] = convert(sampletype, fs)
        for j ∈ 1:config.ndata, k ∈ 1:config.nchannels
            Xs[j][:,:,k,i] = config.preprocess(signal(x[:,k], fs)) |> 
                             s -> rand_padsegment(s, n, config.padsegment) |>
                             s -> config.augment(s) |> 
                             s -> tospec(s, config) |>
                             spec -> freqmaxpool(spec, config.newdims[1])
                             #spec -> freqmaxpool_padsegment(spec, config.newdims; type=config.padsegment)#imresize(spec, config.newdims...)
        end
    end
    return ((Xs, timesec, samplingrates), map(y -> getobs(y, ids), data[2:end])...)#map(Base.Fix2(_getobs, ids), data[2:end])...)#
end

function rand_padsegment(x::AbstractVector{T}, winsize::Int, type::Symbol) where {T}
    wavlen = length(x)
    if wavlen < winsize
        wavlen == winsize && (return x)
        npad = winsize - wavlen
        nleftpad = if type == :random
            rand(1:npad)
        elseif type == :center
            npad ÷ 2
        else
            throw(ArgumentError("Invalid padsegment type"))
        end  
        nrightpad = npad - nleftpad
        return signal([zeros(sampletype, nleftpad);
                       x;
                       zeros(sampletype, nrightpad)], framerate(x))
    elseif wavlen > winsize
        nextra = wavlen - winsize
        startind = if type == :random
            rand(1:nextra)
        elseif type == :center
            ceil(Int, nextra/2)
        else
            throw(ArgumentError("Invalid padsegment type"))
        end
        return signal(x[(1+startind):(startind+winsize)], framerate(x))
    else
        signal(x, framerate(x))
    end
end
function rand_padsegment(x::AbstractVector{T}, config::TSConfig) where {T}
    rand_padsegment(x, config.winsize, config.padsegment)
end
# function rand_segment(x::AbstractVector{T}, config::TSConfig) where {T}
#     wavlen = length(x)
#     if wavlen ≤ config.winsize
#         wavlen == config.winsize && (return x)
#         npad = config.winsize - wavlen
#         nleftpad = config.randsegment ? rand(1:npad) : npad ÷ 2
#         nrightpad = npad - nleftpad
#         return signal([zeros(sampletype, nleftpad, config.nchannels);
#                        x[:,1:config.nchannels];
#                        zeros(sampletype, nrightpad, config.nchannels)], framerate(x))
#     else # wavlen > config.winsize
#         nextra = wavlen - config.winsize
#         startind = config.randsegment ? rand(1:nextra) : ceil(Int, nextra/2)
#         return x[(1+startind):(startind+config.winsize)]
#     end
# end


# function wavread_process(wavpath::AbstractString, config::TSConfig)
#     wavlen = first(wavread(wavpath; format="size"))::Int
#     if wavlen ≤ config.winsize
#         x, fs = wavread(wavpath; format="native")
#         x = convert.(sampletype, x) 
#         wavlen == config.winsize && (return x[:,1:config.nchannels])
#         npad = config.winsize - wavlen
#         nleftpad = config.randsegment ? rand(1:npad) : npad ÷ 2
#         nrightpad = npad - nleftpad
#         return [zeros(sampletype, nleftpad, config.nchannels);
#                 x[:,1:config.nchannels];
#                 zeros(sampletype, nrightpad, config.nchannels)]
#     else # wavlen > config.winsize
#         nextra = wavlen - config.winsize
#         startind = config.randsegment ? rand(1:nextra) : ceil(Int, nextra/2)
#         x = wavread(wavpath, subrange=(1+startind):(startind+config.winsize); format="native") |>
#             first |>
#             a -> convert.(sampletype, a)
#         return x[:,1:config.nchannels]
#     end
# end

"""
Transform a vector of time-series data to a scaled spectrogram specified by `config`.

# Arguments
- x : time-series audio data
- config: SpecConfig instance
- fs : sampling rate

# Returns
- a scaled spectrogram specified by `config`
"""
function tospec(x::SignalAnalysis.SampledSignal, config::SpecConfig)
    stft(x, config.winsize, config.noverlap, DSP.Periodograms.PSDOnly(); window=config.window, fs=framerate(x)) |>
    S -> config.scaled(S, framerate(x)) |>
    sS -> convert.(sampletype, sS)
end

"""
Unpack audio data into a tuple of audio data, time lengths, sampling rates 
by selecting the first augmented view.

# Arguments
- data : acoustic data with multiple augmented views
- withtl : if `true`, includes time lengths (in seconds) of audio files in seconds 
- withsr : if `true`, includes sampling rates of audio files in samples

# Returns
- audio data in the form of first augmented view of the acoustic data and the corresponding time 
lengths and sampling rates   
"""
function unpack_data(data, withtl, withsr)
    ds = first(data)
    iswithtl, normalisetl = withtl
    iswithsr, normalisesr = withsr
    if iswithtl && iswithsr
        first(first(ds)), normalisetl.(ds[2]), normalisesr.(ds[3])
    elseif iswithtl
        first(first(ds)), normalisetl.(ds[2])
    elseif iswithsr
        first(first(ds)), normalisesr.(ds[3])
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
    if d.shuffle 
        @warn "The shuffled targets only corresponds to the latest iteration. Reiterate the " *
              "first item will shuffle targets again."
        targets = Tuple([target[d.indices] for target ∈ targets])
    end
    n == 1 ? only(targets) : targets
end