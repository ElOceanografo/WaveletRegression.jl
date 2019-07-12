
module WaveletRegression

using Wavelets
using Statistics
using LinearAlgebra

export
    wavelm,
    AbstractWaveletTransform,
    MODWT,
    DWT,
    DWPT,
    WaveletRegressionModel,
    modwt_matrix,
    imodwt_matrix,
    dwt_matrix,
    idwt_matrix,
    predict,
    predict_wavespace,
    coef,
    ncoef,
    dof,
    nlevels,
    residuals,
    regression_stderr,
    coef_stderr,
    coef_shrunk,
    shrink

export wavelet, WT

abstract type AbstractWaveletTransform end

for transformtype in [:MODWT, :DWT, :WPT]
    @eval begin
        struct $transformtype <: AbstractWaveletTransform
            wavelet::DiscreteWavelet
            nlevels::Integer
        end
        Wavelets.wavelet(trans::$(transformtype)) = trans.wavelet
        nlevels(trans::$(transformtype)) = trans.nlevels
    end
end

struct WaveletRegressionModel
    X::AbstractArray
    Y::AbstractArray
    n::Integer
    mx::Integer
    my::Integer
    Xw::AbstractArray
    Yw::AbstractArray
    trans::AbstractWaveletTransform
    coefficients
end
Wavelets.wavelet(mod::WaveletRegressionModel) = wavelet(mod.trans)
nlevels(mod::WaveletRegressionModel) = nlevels(mod.trans)
coef(mod::WaveletRegressionModel) = mod.coefficients
ncoef(mod::WaveletRegressionModel) = sum(length(B) for B in mod.coefficients)
dof(mod::WaveletRegressionModel) = mod.n - ncoef(mod) - 1

function predict_wavespace(mod::WaveletRegressionModel, newXw::AbstractArray, shrink=false)
    n = size(newXw, 1)
    if shrink
        BB = coef_shrunk(mod)
    else
        BB = mod.coefficients
    end
    Yw = cat([reshape(newXw[:, j, :] * BB[j], (n, 1, mod.my))
        for j in 1:nlevels(mod)+1]..., dims=2)
    return Yw
end

function predict(mod::WaveletRegressionModel, newX::AbstractVecOrMat, shrink=false)
    @assert size(newX, 2) == size(mod.X, 2)
    newXw = modwt_matrix(newX, mod.trans)
    Yw = predict_wavespace(mod, newXw, shrink)
    Y = imodwt_matrix(Yw, mod.trans)
    return Y
end
predict(mod::WaveletRegressionModel, shrink=false) = predict(mod, mod.X, shrink)

function residuals(mod::WaveletRegressionModel, shrink=false)
    Yw_pred = predict_wavespace(mod, mod.Xw, shrink)
    return mod.Yw .- Yw_pred
end

function regression_stderr(mod::WaveletRegressionModel; unbiased=true, shrink=false)
    resid = residuals(mod, shrink)
    s2 = reshape(sum(abs2 , resid, dims=1), (mod.my, :))
    if unbiased
        return s2 / dof(mod)
    else
        return s2 / mod.n
    end
end

"""
Calculate shrinkage factor for regression coefficients.
J.B. Copas (1983). Regression, prediction, and shrinkage. J. Royal Statistical
Society B 45(3), 311-354.  https://www.jstor.org/stable/2345402
"""
function coef_shrunk(mod::WaveletRegressionModel)
    BBshrunk = deepcopy(coef(mod))
    for j in 1:mod.mx
        Xw = mod.Xw[:, j, :]
        n = mod.n
        p = mod.mx * mod.my
        v = n - p - 1
        σ2 = regression_stderr(mod)[:, j]
        B = coef(mod)[j]
        V = Xw'Xw/n
        M = cholesky(V).U

        shrinkage = zeros(1, mod.my)
        for k in 1:mod.my
            β = B[:, k]
            ξ = M * β
            shrinkage[k] = 1 - ((p-2) * σ2[k] * v) / (n * (v+2) * ξ'ξ)
        end
        BBshrunk[j] = B .* shrinkage
    end
    return BBshrunk
end

function shrink(mod::WaveletRegressionModel)
    BB = coef_shrunk(mod)
    return WaveletRegressionModel(mod.X, mod.Y, mod.n, mod.mx, mod.my,
        mod.Xw, mod.Yw, mod.trans, BB)
end

function coef_stderr(mod::WaveletRegressionModel; unbiased=true, shrink=false)
    # https://en.wikipedia.org/wiki/Ordinary_least_squares
    s2 = regression_stderr(mod, unbiased=unbiased, shrink=shrink)
    L = mod.trans.nlevels + 1
    Qxx =  [mod.Xw[:, j, :]' * mod.Xw[:, j, :] for j in 1:L]
    SE = [similar(B) for B in coef(mod)]
    for j in 1:L
        for i in 1:mod.my
            SE[j][:, i] .= sqrt.(diag(s2[i, j] .* inv(Qxx[j])))
        end
    end
    return SE
end

function modwt_matrix(X::AbstractVecOrMat, trans::MODWT)
    return cat([modwt(X[:, i], trans.wavelet, trans.nlevels) for i in 1:size(X, 2)]..., dims=3)
end

function imodwt_matrix(Xw::AbstractArray, trans::MODWT)
    n, nlevels, nvars = size(Xw)
    hcat([imodwt(Xw[:, :, i], trans.wavelet) for i in 1:nvars]...)
end

function wavelm(X::AbstractVecOrMat, Y::AbstractVecOrMat, trans::MODWT)
    Xw = modwt_matrix(X, trans)
    Yw = modwt_matrix(Y, trans)
    BB = [Xw[:, j, :] \ Yw[:, j, :] for j in 1:(trans.nlevels+1)]
    n = size(X, 1)
    mx = div(length(X), n)
    my = div(length(Y), n)
    return WaveletRegressionModel(X, Y, n, mx, my, Xw, Yw, trans, BB)
end

function dwt_matrix(X::AbstractVecOrMat, trans::DWT)
    return hcat([dwt(X[:, i], trans.wavelet, trans.nlevels) for i in 1:size(X, 2)]...)
end

function idwt_matrix(Xw::AbstractArray, trans::DWT)
    n, nvars = size(Xw)
    hcat([idwt(Xw[:, i], trans.wavelet, trans.nlevels) for i in 1:nvars]...)
end

function wavelm(X::AbstractVecOrMat, Y::AbstractVecOrMat, trans::DWT)
    Xw = dwt_matrix(X, trans)
    Yw = dwt_matrix(Y, trans)
    n = size(Xw, 1)
    indices = [detailrange(n, j) for j in 1:trans.nlevels]
    push!(indices, 1:detailindex(n, trans.nlevels, 1)-1) # scaling coef indices
    BB = [Xw[ii, :] \ Yw[ii, :] for ii in indices]
    n = size(X, 1)
    mx = div(length(X), n)
    my = div(length(Y), n)
    return WaveletRegressionModel(X, Y, n, mx, my, Xw, Yw, trans, BB)
end


end # module
