f(x, b, p, α, μ, σ) = abs(x-b)^(p-1) * pdf(Aepd(μ, σ, p, α), x)

function quantconvert(q, p, α, μ, σ)
    a₁ = quadgk(x -> f(x, q, p, α, μ, σ), -Inf, Inf)[1]
    a₂ = quadgk(x -> f(x, q, p, α, μ, σ), -Inf, q)[1]
    1/((maximum([a₁/a₂, 1.0001]) - 1)^(1/p) + 1)
end

"""
    mcτ(τ, α, p, σ, n, N)

Estimates the converted Lp-quantile
```math
\\tilde{\\tau} = \\left[ \\left(\\frac{E|Y - q_\\tau(1)|^{p-1}}{E |Y-q_\\tau(1)|^{p-1}I\\big(Y \\leq q_\\tau(1)\\big)} -
1\\right)^{\\frac{1}{p}} + 1 \\right]^{-1}
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
        res[i] = quantconvert(q[1], θ, α, 0, σ)
    end
    mean(res)
end
