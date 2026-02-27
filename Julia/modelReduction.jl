# Model reduction as a limiting case of LS-realization.
using HomotopyContinuation
using LinearAlgebra
include("auxiliaryFunctions.jl")

# Four disk system:
d = [0.0448, 0.2368, 0.0013, 0.0211, 0.2250, 0.0219]
c = [1, -1.2024, 2.3675, -2.0039, 2.2337, -1.0420, 0.8513]

# Get impulse response:
using DSP
impulse = [1.0; zeros(99)]
h = filt(d, c, impulse)

# Solve LS realization problem using truncated impulse response:
N = 5 # 5, 10, 15, 30
y = complex(h[1:N])
r = 2
we = "unit"

# Solve using Walsh
sys = Walsh.construct_walsh(y, r; weights=we)
res = solve(sys, only_non_zero=false, threading=true, compile=true)
tab = Walsh.analyze_solutions(y, r, res; system="walsh", weights=we, only_real=true)

# Create contourplot
using Plots

w = Walsh.get_weights(we, N, r); Λ = diagm(w);
f(a0,a1) = norm(sqrt.(Λ)*(y-Walsh.compute_optimal_y_hat(y,ComplexF64.([a0,a1,1]); weights=we)),2)^2;
a1 = range(-5, 5, length=100); # x-axis
a0 = range(-3, 3, length=100); # y-axis
z = zeros(length(a1),length(a0));
for i in eachindex(a1)
    for j in eachindex(a0)
        z[j,i] = f(a0[j], a1[i]);
    end
end
contour(a1, a0, z)

# Scatter real-valued critical points:
xs = [real(v[2]) for v in tab.a]
ys = [real(v[1]) for v in tab.a]
scatter!(xs, ys, label="", markercolor=:red)

# Depict region of stable reduced-order models
tri_x = [-2, 0, 2, -2]; tri_y = [1, -1, 1, 1];
plot!(tri_x, tri_y, linecolor=:black, linewidth=1, label="")
plot!(xlims=(-2.5, 2.5), ylims=(-1.5, 1.5))





# ---------------- Auxiliary to create Tikz figures --------- 

# Solve using FONC
sys = Walsh.construct_fonc(y, r; weights=we)
res = solve(sys, only_non_zero=false, threading=true, compile=true)
tab = Walsh.analyze_solutions(y, r, res; system="walsh", weights=we, only_real=true)

# Output to tikz format:
filtered = [[real(v[1]), real(v[2])] for v in tab.a 
            if abs(real(v[1])) < 1.5 && abs(real(v[2])) < 2.5]
for v in filtered
    println("\\node[red, thick, scale = 1.5] at (axis cs:$(round(v[2], digits=3)),$(round(v[1], digits=3))) {\\pgfuseplotmark{triangle}};")
end