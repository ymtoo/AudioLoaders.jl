import Base.@kwdef

export TSConfig, SpecConfig

abstract type Config end

@kwdef struct TSConfig <: Config
    winsize::Int
    randsegment::Bool
    augment::Function
    nchannels::Int
    ndata::Int
end
TSConfig(winsize, randsegment) = TSConfig(winsize, 
                                          randsegment, 
                                          identity,
                                          1, 
                                          1)

@kwdef struct SpecConfig{W} <: Config
    winsize::Int
    noverlap::Int
    window::W
    scaled::Function
    augment::Function
    newdims::NTuple{2,Int}
    nchannels::Int
    ndata::Int
end
SpecConfig(winsize, noverlap, window) = SpecConfig(winsize, 
                                                   noverlap, 
                                                   window, 
                                                   a -> abs2.(a),
                                                   identity,
                                                   (100,100),
                                                   1,
                                                   1)
