using TimeScaleModification

export Identity, Amplify, PolarityInverse, CircularShift, 
       TimeStretch, PitchShift, BackgroundNoise, TimeMask,
       FrequencyShift

export apply, random_apply

abstract type TSAugmentor end

struct Identity <: TSAugmentor end

function apply(::Identity, x::AbstractVector{T}) where {T}
    x
end

struct Amplify{T<:UnivariateDistribution{Continuous},R<:AbstractRNG} <: TSAugmentor
    ampdist::T
    rng::R
end
Amplify(ampdist::T) where {T<:UnivariateDistribution{Continuous}} = Amplify(ampdist, GLOBAL_RNG)
Amplify(amin::Real, amax::Real) = Amplify(Uniform(amin, amax))
Amplify(amin::Real, amax::Real, rng::R) where {R<:AbstractRNG} = Amplify(Uniform(amin, amax), rng)

function apply(op::Amplify, x::AbstractVector{T}) where {T<:AbstractFloat}
    T(rand(op.rng, op.ampdist)) .* x
end
function apply(op::Amplify, x::AbstractVector{T}) where {T<:Integer}
    rand(op.rng, op.ampdist) .* x
end

struct PolarityInverse{R<:AbstractRNG} <: TSAugmentor 
    rng::R
end
PolarityInverse() = PolarityInverse(GLOBAL_RNG)

function apply(::PolarityInverse, x::AbstractVector{T}) where {T}
    -x
end

struct CircularShift{T<:UnivariateDistribution{Discrete},R<:AbstractRNG} <: TSAugmentor 
    shiftdist::T
    rng::R
end
CircularShift(shiftdist::T) where {T<:UnivariateDistribution{Discrete}} = CircularShift(shiftdist, GLOBAL_RNG)
CircularShift(p::Real) = CircularShift(Geometric(p))
CircularShift(p::Real, rng::R) where {R<:AbstractRNG} = CircularShift(Geometric(p), rng)

function apply(op::CircularShift, x::AbstractVector{T}) where {T}
    shift = rand(op.rng, op.shiftdist)::Int
    shift < 1 && (shift = 1)
    shift ≥ length(x) && (shift = length(x) - 1) 
    circshift(x, shift)
end

struct TimeStretch{T<:UnivariateDistribution{Continuous},M,R<:AbstractRNG} <: TSAugmentor
    sdist::T
    tsmodifier::M
    rng::R
end
TimeStretch(sdist::T, tsmodifier::M) where {T<:UnivariateDistribution{Continuous},M} = TimeStretch(sdist, tsmodifier, GLOBAL_RNG)
TimeStretch(σ::T) where {T<:Real} = TimeStretch(truncated(Normal(one(T), σ), one(T)-3σ, one(T)+3σ),
                                                WSOLA(256, 128, hanning, 10))
TimeStretch(σ::T, rng::R) where {T<:Real,R<:AbstractRNG} = TimeStretch(truncated(Normal(one(T), σ), one(T)-3σ, one(T)+3σ),
                                                                       WSOLA(256, 128, hanning, 10), rng)

function apply(op::TimeStretch, x::AbstractVector{T}) where {T}
    s = rand(op.rng, op.sdist)
    timestretch(op.tsmodifier, x, s)
end

struct PitchShift{T<:UnivariateDistribution{Continuous},M,R<:AbstractRNG} <: TSAugmentor
    sdist::T
    tsmodifier::M
    rng::R
end
PitchShift(sdist::T, tsmodifier::M) where {T<:UnivariateDistribution{Continuous},M} = PitchShift(sdist, tsmodifier, GLOBAL_RNG)
PitchShift(σ::T) where {T<:Real} = PitchShift(truncated(Normal(zero(T), σ), -3σ, 3σ),
                                              WSOLA(256, 128, hanning, 10))
PitchShift(σ::T, rng::R) where {T<:Real,R<:AbstractRNG} = PitchShift(truncated(Normal(zero(T), σ), -3σ, 3σ),
                                                                     WSOLA(256, 128, hanning, 10), rng)
                            

function apply(op::PitchShift, x::AbstractVector{T}) where {T}
    semitones = rand(op.rng, op.sdist)
    pitchshift(op.tsmodifier, x, semitones)
end

struct BackgroundNoise{T,R<:AbstractRNG} <: TSAugmentor
    snrs::Union{Tuple,Real}
    dist::T
    rng::R
end
BackgroundNoise(snrs::Union{Tuple,Real}, dist::T) where {T} = BackgroundNoise(snrs, dist, GLOBAL_RNG)
BackgroundNoise() = BackgroundNoise(0, PinkGaussian)
BackgroundNoise(snrs::Union{Tuple,Real}) = BackgroundNoise(snrs, PinkGaussian)
BackgroundNoise(snrs::Union{Tuple,Real}, rng::R) where {R<:AbstractRNG} = BackgroundNoise(snrs, PinkGaussian, rng)

function apply(op::BackgroundNoise, x::AbstractVector{T}) where {T}
    snr = op.snrs isa Tuple ? rand(op.rng, Uniform(op.snrs...)) : op.snrs |> T
    noise = T.(rand(op.rng, op.dist(length(x))))
    pnoise = mean(abs2, noise)
    a = sqrt((2 * pnoise) / (10 ^ (snr / 10))) |> T
    a .* noise .+ x
end

struct TimeMask{T,R<:AbstractRNG} <: TSAugmentor
    lengthrange::Tuple{Int,Int}
    scaledist::T
    rng::R
end
TimeMask(lengthrange, scaledist) = TimeMask(lengthrange, scaledist, GLOBAL_RNG)
TimeMask(lengthmin, lengthmax, scalemin, scalemax) = TimeMask((lengthmin, lengthmax), 
                                                              Uniform(scalemin, scalemax),
                                                              GLOBAL_RNG)
                                                
function apply(op::TimeMask, x::AbstractVector{T}) where {T}
    length(x) < maximum(op.lengthrange) && throw(ArgumentError("Length of mask is greater than length of `x`")) 
    l = rand(op.rng, range(op.lengthrange...))
    s = rand(op.rng, op.scaledist)
    start = rand(op.rng, 1:(length(x) - l + 1))
    mask = (one(T) - s) .+ s .* (cos.(2π .* range(0,l-one(T)) ./ (l-one(T))) .+ one(T)) ./ 2
    y = copy(x)
    y[start:start+l-1] .*= mask
    y
end

struct FrequencyShift{T<:UnivariateDistribution{Continuous},R<:AbstractRNG} <: TSAugmentor
    shiftdist::T
    rng::R
end
FrequencyShift(σ::Number, rng::R) where {R<:AbstractRNG} = FrequencyShift(Normal(0, σ), rng)
FrequencyShift(σ::Number) = FrequencyShift(Normal(0, σ), GLOBAL_RNG)

"""
Frequency shift of `x`.

Reference
https://gist.github.com/lebedov/4428122
"""
function apply(op::FrequencyShift, x::AbstractVector{T}) where {T}
    shift = convert(T, rand(op.rng, op.shiftdist))
    dt = T(1e-3)
    N = length(x)
    Np = nextpow(2, N)
    t = range(0, length=Np)
    xpad = [x;zeros(T,Np-N)]
    real((hilbert(xpad) .* exp.(T(2)im * π * shift * dt * t))[1:N])
end

function random_apply(op::TSAugmentor, x; p=1.0)
    if rand(op.rng) < p
        apply(op, x)
    else
        x
    end
end
