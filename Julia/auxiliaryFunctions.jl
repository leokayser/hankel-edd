module Walsh

# ------ LOAD PACKAGES ---------
using LinearAlgebra
using HomotopyContinuation
import PolynomialRoots
import TypedTables
using Random
using ToeplitzMatrices

# ------ AUXILIARY FUNCTIONS --------
function get_weights(weights, N::Int, r::Int)

    if (weights === nothing) || (weights == "unit") # Euclidean metric on R^N
        w = ones(N)
    elseif weights == "frob" # Frobenius for (N-n)x(n+1) Hankel
        d = min(N - r, r + 1)
        w = vec([min(i + 1, d, N - i) for i = 0:N-1])
    elseif weights == "randn" # Random (generic) weights:
        w = randn(N).^2
    elseif weights == "bomb" # Bombieri 
        w = vec([binomial(N - 1, i) for i in 0:N-1])
    elseif weights == "unit_ottaviani" # = Frobenius for square Hankel
        w = vec([min(i + 1, N - i) for i in 0:N-1])
    elseif weights == "sym" # Symbolic weights (p = 1 / λ) 
        @var p[1:N]
        w = inv.(p)
    else
        w = weights # Manually submitted weights:
    end

    return w
end

function construct_walsh(y::Vector{ComplexF64}, r::Int; weights=nothing)
    N = length(y)

    w = get_weights(weights, N, r)
    Λ = diagm(w)
    println("Initialized weights as: $(weights)")

    # Compose symbolic system of equations:
    @var a[0:r]
    T = zeros(Expression, N - r, N)
    for i in 1:N-r
        T[i, i:i-1+length(a)] = a
    end
    S = zeros(Expression, N - 2r, N - r)
    for i in 1:N-2r
        S[i, i:i-1+length(a)] = a
    end
    @var g[1:N-2r]
    system = Vector{Expression}(undef, N - r + 1)
    system[1:N-r] = T * y - T * inv(Λ) * transpose(T) * transpose(S) * g
    system[N-r+1] = a[r+1] - 1 # Dehomogenization

    if weights == "sym"
        P = System(system, variables=[a; g], parameters=variables(w))
    else
        P = System(system, variables=[a; g])
    end

    return P
end

function construct_fonc(y::Vector{ComplexF64}, r::Int; weights=nothing)
    N = length(y)

    w = get_weights(weights, N, r)
    Λ = diagm(w)

    @var a[0:r]
    T_a = zeros(Expression, N - r, N)
    for i in 1:N-r
        T_a[i, i:i+length(a)-1] = a
    end

    @var l[0:N-r-1]
    T_l = zeros(Expression, r + 1, N)
    for i in 1:r+1
        T_l[i, i:i+length(l)-1] = l
    end

    system = Vector{Expression}(undef, N + 2)
    system[1:N-r] = T_a * y - T_a * inv(Λ) * transpose(T_a) * l
    system[N-r+1:N+1] = T_l * y - T_l * inv(Λ) * transpose(T_l) * a
    system[N+2] = a[r+1] - 1 # Dehomogenization

    if weights == "sym"
        P = System(system, variables=[a; l], parameters=variables(w))
    else
        P = System(system, variables=[a; l])
    end

    return P
end

function compute_optimal_y_hat(y::Vector{ComplexF64}, a::Vector{ComplexF64}; weights=nothing)
    N = length(y)
    r = length(a) - 1

    w = get_weights(weights, N, r)
    Λ = diagm(w)

    T = zeros(ComplexF64, N - r, N)
    for i in 1:N-r
        T[i, i:i+length(a)-1] = a
    end

    # Compute orthogonal projection:
    y_hat = (I - inv(Λ)* T' * inv(T * inv(Λ) * T') * T) * y

    return y_hat
end

function analyze_solutions(y::Vector{ComplexF64}, r::Int, res::Result; only_real=false, system=nothing, weights=nothing, only_finite=true, multiple=true)

    # Extract solutions:
    sol = solutions(res, only_nonsingular=false, multiple_results=multiple, only_finite=only_finite, only_real=only_real)

    # Parse model parameters:
    models = Vector(undef, length(sol))
    other_vars = Vector(undef, length(sol))
    for i in eachindex(sol)
        models[i] = sol[i][1:r+1]
        other_vars[i] = sol[i][r+2:end]
    end

    N = length(y)
    w = get_weights(weights, N, r)
    Λ = diagm(w)

    # Analyze solutions
    cost = Vector{Float64}(undef, length(models))
    y_hats = Vector{Vector{ComplexF64}}(undef, length(models))
    sings = Vector{Vector{Float64}}(undef, length(models))
    poles = Vector{Vector}(undef, length(models))
    for i in range(1, length(models))
        #display("------------------")
        a = models[i] # a_0 ... a_r
        poles[i] = PolynomialRoots.roots(a)
        y_hats[i] = compute_optimal_y_hat(y, a; weights=weights)
        cost[i] = norm(sqrt.(Λ) * (y - y_hats[i]), 2)^2

        # Singular values of Yhat
        Y_hat = Hankel(y_hats[i], (N - r, r + 1))
        sings[i] = vec(svdvals(Y_hat))
    end

    # Sort results on increasing cost function value:
    order = sortperm(cost)

    # Summarize solutions in table:
    if system == "lagrangian"
        tab = TypedTables.Table(i=order, f2=cost[order], f=sqrt.(cost[order]), a=models[order], p=poles[order], l=other_vars[order], yhat=y_hats[order], sings=sings[order])
    elseif system == "walsh"
        tab = TypedTables.Table(i=order, f2=cost[order], f=sqrt.(cost[order]), a=models[order], p=poles[order], g=other_vars[order], yhat=y_hats[order], sings=sings[order])
    end

    return tab
end

end