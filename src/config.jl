import Base.@kwdef

export TSConfig, SpecConfig

abstract type Config end

@kwdef struct TSConfig <: Config
    winsize::Int
    randsegment::Bool
    preprocess_augment::Function
    nchannels::Int
    ndata::Int
end
TSConfig(winsize, randsegment) = TSConfig(winsize, 
                                          randsegment, 
                                          (x, fs) -> identity(x),
                                          1, 
                                          1)

@kwdef struct SpecConfig{W} <: Config
    winsize::Int
    noverlap::Int
    window::W
    scaled::Function
    preprocess_augment::Function
    newdims::NTuple{2,Int}
    nchannels::Int
    ndata::Int
end
SpecConfig(winsize, noverlap, window) = SpecConfig(winsize, 
                                                   noverlap, 
                                                   window, 
                                                   a -> abs2.(a),
                                                   (x, fs) -> identity(x),
                                                   (100,100),
                                                   1,
                                                   1)
