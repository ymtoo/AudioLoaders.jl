export embed

"""
Embed either time-series or scaled spectrogram of audio files using a function `f`. 

# Argments
- f : embedding function
- d : AudioLoader instance
- withtl : Tuple with first entry indicating whether time lengths (in seconds) of audio files are included and  
           the second entry is a function to normalize the time length.
- withsr : Tuple with first entry indicating whether sampling rates (in samples) of audio files are included and
           the second entry is a function to normalize the sampling rate.
- todevice : move data to `todevice`
- showprogress : if `true`, show the progress

# Returns
- a matrix with each column represents an embedding vector of an audio file 
"""
function embed(f::Function, 
               d::AudioLoader; 
               withtl::Tuple{Bool,Function}=(true,identity),
               withsr::Tuple{Bool,Function}=(true,identity), 
               todevice::Function=cpu, 
               showprogress::Bool=false)
    p = Progress(length(d); enabled=showprogress)
    n = length(first(d.data))
    z = f(todevice.(unpack_data(first(d), withtl, withsr))...)
    m, batchsize = size(z)
    Z = zeros(sampletype, m, n)
    Z[:,1:batchsize] = cpu(z)
    for (i, d1) ∈ enumerate(d)
        if i > 1
            z = f(todevice.(unpack_data(d1, withtl, withsr))...)
            startindex = (i - 1) * batchsize + 1
            stopindex = startindex + size(z, 2) - 1
            Z[:,startindex:stopindex] = cpu(z) 
        end
        next!(p)
    end
    Z
end