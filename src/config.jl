import Base.@kwdef

export TSConfig, SpecConfig

abstract type Config end

@kwdef struct TSConfig <: Config
    winsize::Int
    preprocess_augment::Function
    nchannels::Int
    ndata::Int
    padsegment::Symbol
end
TSConfig(winsize) = TSConfig(winsize, 
                             x -> identity(x),
                             1, 
                             1,
                             :center)

@kwdef struct SpecConfig{W} <: Config
    winsize::Int
    noverlap::Int
    window::W
    scaled::Function
    preprocess_augment::Function
    newdims::NTuple{2,Int}
    nchannels::Int
    ndata::Int
    padsegment::Symbol
end
SpecConfig(winsize, noverlap, window) = SpecConfig(winsize, 
                                                   noverlap, 
                                                   window, 
                                                   (a, nfft, fs) -> melscale(a, nfft; fs=fs),
                                                   x -> identity(x),
                                                   (100,100),
                                                   1,
                                                   1,
                                                   :center)
