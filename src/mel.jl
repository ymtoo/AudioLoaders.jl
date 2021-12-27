export hz2mel, mel2hz
export fft_frequencies, mel_frequencies
export getmelfilters, melspectrogram, melscale

"""
Mel-spectrogram.
"""
function melscale(S::AbstractMatrix{T},
                  nfft::Int; 
                  nmels::Int = 128, 
                  fs::Real=1,
                  kwargs...) where {T}
    getmelfilters(fs, nfft, nmels, kwargs...) * S
end
function melspectrogram(x::AbstractVector{T}, 
                        nfft::Int, 
                        noverlap::Int; 
                        nmels::Int = 128, 
                        fs::Real=1,
                        window::Union{Function,AbstractVector,Nothing}=nothing,
                        kwargs...) where {T}
    S = stft(x, nfft, noverlap, DSP.Periodograms.PSDOnly(); fs=fs, window=window)
    melscale(S; nmels=nmels, fs=fs, kwargs...)
end

"""
Create a Mel filter-bank.

The implementation is based on 
1. https://librosa.org/doc/main/_modules/librosa/filters.html#mel
2. https://github.com/JuliaMusic/MusicProcessing.jl/blob/master/src/mel.jl
"""
function getmelfilters(fs::T, 
                       nfft::Int, 
                       nmels::Int; 
                       fmin::Real=zero(T), 
                       fmax=fs/2) where {T<:Real}
    weights = zeros(nmels, 1 + nfft ÷ 2)
    fftfreqs = fft_frequencies(fs, nfft)
    melfreqs = mel_frequencies(nmels + 2, fmin, fmax)

    # slaney-style mel
    enorm = 2 ./ (melfreqs[3:end] .- melfreqs[1:nmels])

    for i ∈ 1:nmels
        lower = (fftfreqs .- melfreqs[i]) ./ (melfreqs[i+1] - melfreqs[i]) 
        upper = (melfreqs[i+2] .- fftfreqs) ./ (melfreqs[i+2] - melfreqs[i+1])
        weights[i,:] = max.(0, min.(lower, upper)) .* enorm[i]
    end
    return weights
end

"""
Return DFT sample frequencies.
"""
fft_frequencies(fs::Real, nfft::Int) = range(start=0, stop=fs/2, length=1+nfft÷2) |> collect

"""
Compute mel scale.
"""
function mel_frequencies(nmels::Int, fmin::Real, fmax::Real)
    minmel = hz2mel(fmin)
    maxmel = hz2mel(fmax)

    mels = range(start=minmel, stop=maxmel, length=nmels) |> collect
    mel2hz.(mels)
end

"""
Convert Hz to Mel.
"""
function hz2mel(freq::T) where {T<:Real}
    fmin = 0
    fsp = 200 / 3

    mel = (freq - fmin) / fsp
    
    minloghz = 1000
    minlogmel = (minloghz - fmin) / fsp
    logstep = log.(6.4) / 27

    if freq ≥ minloghz
        mel = minlogmel + log(freq / minloghz) / logstep
    end
    return mel
end

"""
Convert mel bin number to frequency.
"""
function mel2hz(mel)
    fmin = 0
    fsp = 200 / 3
    freq = fmin + fsp * mel

    minloghz = 1000
    minlogmel = (minloghz - fmin) / fsp
    logstep = log(6.4) / 27

    if mel ≥ minlogmel
        freq = minloghz * exp(logstep * (mel - minlogmel))
    end
    return freq
end