module WaveletRegression

using Wavelets
using Statistics
export wavelm, WaveletRegressionModel, modwt_matrix, imodwt_matrix, predict
export wavelet, WT

struct WaveletRegressionModel
    X
    Y
    Xw
    Yw
    wt
    nlevels
    coefficients
    Xintercept
    Yintercept
end

function predict(mod::WaveletRegressionModel, newX)
    @assert size(newX) == size(mod.X)
    newXw = modwt_matrix(newX, mod.wt, mod.nlevels)
    n, my = size(mod.Y)
    Yw = cat([reshape(newXw[:, j, :] * mod.coefficients[j], (n, 1, my))
        for j in 1:mod.nlevels+1]..., dims=2)
    Y = imodwt_matrix(Yw, mod.wt)
    return Y
end
predict(mod::WaveletRegressionModel) = predict(mod, mod.X)

function modwt_matrix(X, wt, nlevels)
    return cat([modwt(X[:, i], wt, nlevels) for i in 1:size(X, 2)]..., dims=3)
end

function imodwt_matrix(Xw, wt)
    n, nlevels, nvars = size(Xw)
    hcat([imodwt(Xw[:, :, i], wt) for i in 1:nvars]...)
end

function wavelm(X::AbstractVecOrMat, Y::AbstractVecOrMat, wt, nlevels)
    Xintercept = mean(X, dims=1)
    Yintercept = mean(Y, dims=1)
    Xw = modwt_matrix(X, wt, nlevels)
    Yw = modwt_matrix(Y, wt, nlevels)
    BB = [Xw[:, j, :] \ Yw[:, j, :] for j in 1:(nlevels+1)]
    return WaveletRegressionModel(X, Y, Xw, Yw, wt, nlevels, BB, Xintercept, Yintercept)
end


end # module
