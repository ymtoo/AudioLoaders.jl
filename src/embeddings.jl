export embed

function embed(f::Function, d::AudioLoader; todevice::Function=cpu, showprogress::Bool=false)
    p = Progress(length(d); enabled=showprogress)
    n = length(first(d.data))
    z = f(todevice.(unpack_data(d.config, first(d)))...)
    m, batchsize = size(z)
    Z = zeros(sampletype, m, n)
    Z[:,1:batchsize] = cpu(z)
    for (i, d1) ∈ enumerate(d)
        if i > 1
            z = f(todevice.(unpack_data(d.config, d1))...)
            startindex = (i - 1) * batchsize + 1
            stopindex = startindex + size(z, 2) - 1
            Z[:,startindex:stopindex] = cpu(z) 
        end
        next!(p)
    end
    Z
end