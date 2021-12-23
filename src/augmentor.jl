using TimeScaleModification

export Identity, Amplify, PolarityInverse, CircularShift, 
       TimeStretch, PitchShift, BackgroundNoise
export apply, random_apply

abstract type TSAugmentor end

struct Identity <: TSAugmentor end

function apply(::Identity, x::AbstractVector{T}) where {T}
    x
end

struct Amplify{T<:UnivariateDistribution{Continuous}} <: TSAugmentor
    ampdist::T
end
Amplify(amin::Real, amax::Real) = Amplify(Uniform(amin, amax))

function apply(op::Amplify, x::AbstractVector{T}) where {T<:AbstractFloat}
    T(rand(op.ampdist)) .* x
end
function apply(op::Amplify, x::AbstractVector{T}) where {T<:Integer}
    rand(op.ampdist) .* x
end

struct PolarityInverse <: TSAugmentor end

function apply(::PolarityInverse, x::AbstractVector{T}) where {T}
    -x
end

struct CircularShift{T<:UnivariateDistribution{Discrete}} <: TSAugmentor 
    shiftdist::T
end
CircularShift(p::Real) = CircularShift(Geometric(p))

function apply(op::CircularShift, x::AbstractVector{T}) where {T}
    shift = rand(op.shiftdist)::Int
    shift < 1 && (shift = 1)
    shift ≥ length(x) && (shift = length(x) - 1) 
    circshift(x, shift)
end

struct TimeStretch{T<:UnivariateDistribution{Continuous},M} <: TSAugmentor
    sdist::T
    tsmodifer::M
end
TimeStretch(σ::T) where {T<:Real} = TimeStretch(truncated(Normal(one(T), σ), one(T)-3σ, one(T)+3σ),
                                                WSOLA(256, 128, hanning, 10))
function apply(op::TimeStretch, x::AbstractVector{T}) where {T}
    s = rand(op.sdist)
    timestretch(op.tsmodifer, x, s)
end

struct PitchShift{T<:UnivariateDistribution{Continuous},M} <: TSAugmentor
    sdist::T
    tsmodifer::M
end
PitchShift(σ::T) where {T<:Real} = PitchShift(truncated(Normal(zero(T), σ), -3σ, 3σ),
                                              WSOLA(256, 128, hanning, 10))

function apply(op::PitchShift, x::AbstractVector{T}) where {T}
    semitones = rand(op.sdist)
    pitchshift(op.tsmodifer, x, semitones)
end

struct BackgroundNoise{T} <: TSAugmentor
    dist::T
    snrs::Union{Tuple,Real}
end
BackgroundNoise() = BackgroundNoise(RedGaussian, 0)

function apply(op::BackgroundNoise, x::AbstractVector{T}) where {T}
    snr = op.snrs isa Tuple ? rand(Uniform(op.snrs...)) : op.snrs |> T
    noise = T.(rand(op.dist(length(x))))
    pnoise = mean(abs2, noise)
    a = sqrt((2 * pnoise) / (10 ^ (snr / 10))) |> T
    a .* noise .+ x
end

function random_apply(op::TSAugmentor, x; p=1.0)
    if rand() < p
        apply(op, x)
    else
        x
    end
end
