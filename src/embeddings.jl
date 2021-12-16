export embed

function embed(f::Function, 
               d::AudioLoader; 
               withtimesec::Bool=true,
               withsamplingrate::Bool=true, 
               todevice::Function=cpu, 
               showprogress::Bool=false)
    p = Progress(length(d); enabled=showprogress)
    n = length(first(d.data))
    z = f(todevice.(unpack_data(first(d), withtimesec, withsamplingrate))...)
    m, batchsize = size(z)
    Z = zeros(sampletype, m, n)
    Z[:,1:batchsize] = cpu(z)
    for (i, d1) ∈ enumerate(d)
        if i > 1
            z = f(todevice.(unpack_data(d1, withtimesec, withsamplingrate))...)
            startindex = (i - 1) * batchsize + 1
            stopindex = startindex + size(z, 2) - 1
            Z[:,startindex:stopindex] = cpu(z) 
        end
        next!(p)
    end
    Z
end