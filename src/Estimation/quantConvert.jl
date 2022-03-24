f(x, b, p, α, μ, σ) = abs(x-b)^(p-1) * pdf(Aepd(μ, σ, p, α), x)

"""
    quantconvert(q, p, α, μ, σ)

Computes
```math
\\tilde{\\tau} = \\left[ \\left(\\frac{E|Y - q_\\tau(1)|^{p-1}}{E |Y-q_\\tau(1)|^{p-1}I\\big(Y \\leq q_\\tau(1)\\big)} -
1\\right)^{\\frac{1}{p}} + 1 \\right]^{-1}
```

# Arguments
- `q::Real`: Quantile coeffcient estimate
- `θ::Real`: Shape of the SEPD
- `α::Real`: Skewness of the SEPD
- `μ::Real`: Location of the SEPD
- `σ::Real`: Scale of the SEPD
"""
function quantconvert(q, θ, α, μ, σ)
    a₁ = quadgk(x -> f(x, q, θ, α, μ, σ), -Inf, Inf)[1]
    a₂ = quadgk(x -> f(x, q, θ, α, μ, σ), -Inf, q)[1]
    1/((maximum([a₁/a₂, 1.00001]) - 1)^(1/p) + 1)
end

"""
    mcτ(τ, α, p, σ, n, N)

Estimates the converted Lp-quantile
```math
\\tilde{\\tau} = \\left[ \\left(\\frac{E|Y - q_\\tau(1)|^{\\theta-1}}{E |Y-q_\\tau(1)|^{\\theta-1}I\\big(Y \\leq q_\\tau(1)\\big)} -
1\\right)^{\\frac{1}{\\theta}} + 1 \\right]^{-1}
```
as a MC-estimate

# Arguments
- `τ::Real`: Quantile level
- `α::Real`: Skewness of the SEPD
- `θ::Real`: Shape of the SEPD
- `σ::Real`: Scale of the SEPD
- `n::Int`: Sample size of generated data
- `N::Int`: Number of replications
"""
function mcτ(τ::Real, α::Real, θ::Real, σ::Real, n::Int = 1000, N::Int = 1000)
    res = zeros(N)
    for i in 1:N
        dat = rand(Aepd(0, σ, θ, α), n)
        q = DataFrame(hcat(dat), :auto) |> x -> qreg(@formula(x1 ~  1), x, τ) |> coef;
        #res[i] = quantconvert(q[1], θ, α, 0, σ)
        a₁ = mean(abs.(dat .- q).^(p-1))
        a₂ = mean(abs.(dat .- q).^(p-1) .* (dat .< q))
        res[i] = 1/((maximum([a₁/a₂, 1.0001]) - 1)^(1/p) + 1)
    end
    mean(res)
end
