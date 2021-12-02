using TimeScaleModification

export Identity, Amplify, PolarityInverse, CircularShift, TimeStretch, PitchShift
export apply, random_apply

abstract type TSAugmentor end

struct Identity <: TSAugmentor end

function apply(::Identity, x::AbstractVector{T}) where {T}
    x
end

struct Amplify <: TSAugmentor
    ampdist::UnivariateDistribution
end
Amplify(amin::Real, amax::Real) = Amplify(Uniform(amin, amax))

function apply(op::Amplify, x::AbstractVector{T}) where {T}
    T(rand(op.ampdist)) .* x
end

struct PolarityInverse <: TSAugmentor end

function apply(::PolarityInverse, x::AbstractVector{T}) where {T}
    -x
end

struct CircularShift <: TSAugmentor 
    shiftdist::UnivariateDistribution
end
CircularShift(p::Real) = CircularShift(Geometric(p))

function apply(op::CircularShift, x::AbstractVector{T}) where {T}
    shift = rand(truncated(op.shiftdist, 1, length(x)-1))
    circshift(x, shift)
end

struct TimeStretch{M} <: TSAugmentor
    sdist::UnivariateDistribution
    tsmodifer::M
end
TimeStretch(σ::T) where {T<:Real} = TimeStretch(truncated(Normal(one(T), σ), one(T)-3σ, one(T)+3σ),
                                                WSOLA(256, 128, hanning, 10))

function apply(op::TimeStretch, x::AbstractVector{T}) where {T}
    s = rand(op.sdist)
    timestretch(op.tsmodifer, x, s)
end

struct PitchShift{M} <: TSAugmentor
    sdist::UnivariateDistribution
    tsmodifer::M
end
PitchShift(σ::T) where {T<:Real} = PitchShift(truncated(Normal(zero(T), σ), -3σ, 3σ),
                                              WSOLA(256, 128, hanning, 10))

function apply(op::PitchShift, x::AbstractVector{T}) where {T}
    semitones = rand(op.sdist)
    pitchshift(op.tsmodifer, x, semitones)
end

function random_apply(op::TSAugmentor, x; p=1.0)
    if rand() > p
        apply(op, x)
    else
        x
    end
end