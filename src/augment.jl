mutatedate(x) = datetime2unix(unix2datetime(x) + Dates.Year(1000))

function reflect(data, s)
    x = getfield(data, s)
    if s == :指数
        x′ = reflect_price(x)
    elseif s == :日期
        x′ = mutatedate.(x)
    elseif s == :特征
        x′ = reflect_feature(x, data.特征列)
    elseif s == :涨幅
        x′ = -x
    elseif s == :买手续费率
        x′ = data.卖手续费率
    elseif s == :卖手续费率
        x′ = data.买手续费率
    elseif s == :涨停
        x′ = data.跌停
    elseif s == :跌停
        x′ = data.涨停
    elseif s == :价格
        x′ = reflect_price(x)
    else
        x′ = deepcopy(x)
    end
    if isa(x, AbstractArray)
        return cat(x, x′, dims = ndims(x))
    else
        return x′
    end
end

reflect(data) = Data([reflect(data, s) for s in fieldnames(Data)]...)

reflect_price(x) = 2 .* x[:, 1:1] .- x

function reflect_feature(x, dict)
    x′ = copy(x)
    for (fea, col) in dict
        if occursin("量", fea) || occursin("额", fea) || occursin("仓", fea)
            x′[col, :,  :] = x[col, :,  :]
        elseif fea == "上滑点"
            x′[col, :,  :] = x[dict["下滑点"], :, :]
        elseif fea == "下滑点"
            x′[col, :,  :] = x[dict["上滑点"], :, :]
        elseif fea == "高"
            x′[col, :,  :] = -x[dict["低"], :, :]
        elseif fea == "低"
            x′[col, :,  :] = -x[dict["高"], :, :]
        else
           x′[col, :,  :] = -x[col, :,  :]
        end
    end
    return x′
end