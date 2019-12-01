function str2int(str)
    s = 0
    for (i, c) in enumerate(reverse(uppercase(str)))
        x = ifelse(c >= 'A', c - 'A' + 10, c - '0')
        s += x * 36^(i - 1)
    end
    UInt32(s)
end

function int2str(x)
    io = IOBuffer()
    for x in reverse(digits(x, base = 36))
        c = ifelse(x >= 10, Char(x - 10 + 65), Char(x + 48))
        write(io, c)
    end
    String(take!(io))
end

flt2str(x::Float32) = int2str(reinterpret(UInt32, x))
flt2str(x) = flt2str(convert(Float32, x))

str2flt(str) = reinterpret(Float32, str2int(str))