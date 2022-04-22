module AudioLoaders

using Reexport

using Distributions
using DSP
@reexport using DSP: Windows, stft, power, pow2db
using MLUtils: numobs, getobs
using Flux: cpu, gpu, MaxPool
#import ImageFiltering: mapwindow
#using ImageTransformations
using ProgressMeter
using SignalAnalysis
using Random: AbstractRNG, shuffle!, GLOBAL_RNG
using WAV

include("mel.jl")
include("config.jl")
include("audioloader.jl")
include("utils.jl")
include("embeddings.jl")
include("augmentor.jl")

const sampletype = Float32

end # module
