export embed

function embed(f::Function, d::AudioLoader; device::Function=cpu)
    n = length(first(d.data))
    z = f(device.(unpack_data(d.config, first(d)))...)
    m, batchsize = size(z)
    Z = zeros(sampletype, m, n)
    Z[:,1:batchsize] = z
    for (i, d1) ∈ enumerate(d)
        if i > 1
            z = f(device.(unpack_data(d.config, d1))...)
            startindex = (i - 1) * batchsize + 1
            stopindex = startindex + size(z, 2) - 1
            Z[:,startindex:stopindex] = cpu(z) 
        end
    end
    Z
end