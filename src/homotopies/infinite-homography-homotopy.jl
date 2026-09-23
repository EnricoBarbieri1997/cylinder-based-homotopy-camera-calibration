export InfiniteHomographyHomotopy, interpolate_line, verify_line_through_vanishing_point, matrix_exp, matrix_log, tp!, get_vanishing_point

using LinearAlgebra: cross, norm, normalize, dot, pinv, I, eigen, Diagonal
using HomotopyContinuation

"""
    line_angle(line)

Compute the angle of a line's normal direction with respect to the x-axis.
For a line `l = [a, b, c]`, returns `atan(b, a)` in radians.
"""
function line_angle(line::AbstractVector)
    return atan(line[2], line[1])
end

"""
    matrix_log(H)

Compute the matrix logarithm of a 3x3 matrix H.
Uses eigendecomposition: if H = V * D * V⁻¹, then log(H) = V * log(D) * V⁻¹.
"""
function matrix_log(H::AbstractMatrix)
    F = eigen(H)
    V = F.vectors
    D = Diagonal(log.(Complex.(F.values)))
    return real.(V * D * inv(V))
end

"""
    matrix_exp(A)

Compute the matrix exponential of a 3x3 matrix A.
"""
function matrix_exp(A::AbstractMatrix)
    F = eigen(A)
    V = F.vectors
    D = Diagonal(exp.(Complex.(F.values)))
    return real.(V * D * inv(V))
end

"""
    InfiniteHomographyHomotopy

A parameter homotopy that interpolates lines across views related by homographies H_∞.

Supports multiple H_∞ matrices, where each line can use a different (H_∞, vanishing_point) pair
via an index mapping.

For each line ℓ₀ passing through vanishing point v₀, and target line ℓ₁ passing through v₁ = H_∞ * v₀,
the homotopy interpolates lines as follows:
1. Compute angle of each starting and target line (angle of normal direction: atan(b, a) for line [a,b,c])
2. Compute angle difference between the starting and target lines (wrapped to [-π, π])
3. Interpolate vanishing point: v_t = H_t * v₀ where H_t = exp(t * log(H_∞))
4. Interpolate line angle: θ_t = θ₀ + t*(θ₁ - θ₀)
5. Reconstruct ℓ_t as line passing through v_t=[vx,vy,vz] with angle θ_t:
   c = -(cos(θ_t)*vx + sin(θ_t)*vy) / vz
   ℓ_t = [cos(θ_t), sin(θ_t), c]

This ensures ℓ_t always passes through v_t = H_t * v₀.
"""
struct InfiniteHomographyHomotopy{T<:AbstractSystem} <: AbstractHomotopy
    F::T
    p::Vector{ComplexF64}  # start parameters (flattened lines)
    q::Vector{ComplexF64}  # target parameters (flattened lines)

    # Arrays of H_inf matrices (one per group)
    H_infs::Vector{Matrix{Float64}}        # H_∞ for each group
    log_H_infs::Vector{Matrix{Float64}}    # log(H_∞) for each group

    # Vanishing points per group
    vanishing_points_per_group::Vector{Vector{Float64}}

    # Line-to-group mapping
    h_indices::Vector{Int}  # h_indices[i] = group index for line i

    # Per-line precomputed data
    angles_start::Vector{Float64}  # Precomputed angles of starting lines
    angles_target::Vector{Float64}  # Precomputed angles of target lines
    angle_diff::Vector{Float64}  # Precomputed angle difference between starting line and target line

    # Cache
    t_cache::Base.RefValue{ComplexF64}
    pt::Vector{ComplexF64}
    taylor_pt::TaylorVector{2,ComplexF64}

    # H_t cache per group (optimization)
    H_t_cache::Vector{Matrix{Float64}}
end

# Named parameters version (supports both legacy single H_inf and new multiple H_infs API)
# Dispatches based on whether H_inf or H_infs keyword is provided
function InfiniteHomographyHomotopy(
    F;
    start_parameters::AbstractVector,
    target_parameters::AbstractVector,
    H_inf::Union{AbstractMatrix, Nothing} = nothing,
    H_infs::Union{AbstractVector{<:AbstractMatrix}, Nothing} = nothing,
    vanishing_points::Union{AbstractVector, Nothing} = nothing,
    vanishing_points_per_group::Union{AbstractVector, Nothing} = nothing,
    h_indices::Union{AbstractVector{Int}, Nothing} = nothing,
)
    # New API: multiple H_infs with indexing
    if H_infs !== nothing && vanishing_points_per_group !== nothing && h_indices !== nothing
        return InfiniteHomographyHomotopy(F, start_parameters, target_parameters, H_infs, vanishing_points_per_group, h_indices)
    end
    # Legacy API: single H_inf with per-line vanishing points
    if H_inf !== nothing && vanishing_points !== nothing
        return InfiniteHomographyHomotopy(F, start_parameters, target_parameters, H_inf, vanishing_points)
    end
    error("Invalid arguments: provide either (H_inf, vanishing_points) or (H_infs, vanishing_points_per_group, h_indices)")
end

# ModelKit.System version (new API)
function InfiniteHomographyHomotopy(
    F::ModelKit.System,
    p::AbstractVector,
    q::AbstractVector,
    H_infs::AbstractVector{<:AbstractMatrix},
    vanishing_points_per_group::AbstractVector,
    h_indices::AbstractVector{Int};
    compile::Union{Bool,Symbol} = true,
)
    InfiniteHomographyHomotopy(fixed(F; compile = compile), p, q, H_infs, vanishing_points_per_group, h_indices)
end

# ModelKit.System version (legacy API)
function InfiniteHomographyHomotopy(
    F::ModelKit.System,
    p::AbstractVector,
    q::AbstractVector,
    H_inf::AbstractMatrix,
    vanishing_points::AbstractVector;
    compile::Union{Bool,Symbol} = true,
)
    InfiniteHomographyHomotopy(fixed(F; compile = compile), p, q, H_inf, vanishing_points)
end

# Legacy API: single H_inf with per-line vanishing points
# Converts to new format by creating one group per line (all sharing the same H_inf)
function InfiniteHomographyHomotopy(
    F::AbstractSystem,
    p::AbstractVector,
    q::AbstractVector,
    H_inf::AbstractMatrix,
    vanishing_points::AbstractVector
)
    number_of_lines = length(p) ÷ 3
    # Each line becomes its own group with the shared H_inf
    H_infs = [Matrix{Float64}(H_inf) for _ in 1:number_of_lines]
    vps_per_group = [Vector{Float64}(vanishing_points[i]) for i in 1:number_of_lines]
    h_indices = collect(1:number_of_lines)
    return InfiniteHomographyHomotopy(F, p, q, H_infs, vps_per_group, h_indices)
end

# New primary API: multiple H_infs with per-line indexing
function InfiniteHomographyHomotopy(
    F::AbstractSystem,
    p::AbstractVector,
    q::AbstractVector,
    H_infs::AbstractVector{<:AbstractMatrix},
    vanishing_points_per_group::AbstractVector,
    h_indices::AbstractVector{Int}
)
    @assert length(p) == length(q) == nparameters(F)

    p̂ = Vector{ComplexF64}(p)
    q̂ = Vector{ComplexF64}(q)
    taylor_pt = TaylorVector{2}(ComplexF64, length(q))
    pt = copy(p̂)

    number_of_lines = length(p) ÷ 3
    num_groups = length(H_infs)

    @assert length(vanishing_points_per_group) == num_groups "Need one vanishing point per H_inf group"
    @assert length(h_indices) == number_of_lines "Need one index per line"
    @assert all(1 .<= h_indices .<= num_groups) "All h_indices must be in [1, $num_groups]"

    # Precompute per-group homography matrices
    H_infs_f = [Matrix{Float64}(H) for H in H_infs]
    log_H_infs = [matrix_log(H) for H in H_infs_f]

    # Normalize vanishing points per group
    vps_per_group = [normalize(Vector{Float64}(v)) for v in vanishing_points_per_group]

    # Initialize H_t cache (will be computed in tp!)
    H_t_cache = [zeros(Float64, 3, 3) for _ in 1:num_groups]

    # Precompute per-line data
    angles_start = zeros(Float64, number_of_lines)
    angles_target = zeros(Float64, number_of_lines)
    angles_diff = zeros(Float64, number_of_lines)

    for i in 1:number_of_lines
        idx = (i-1)*3 + 1
        line_start = real.(p[idx:idx+2])
        line_target = real.(q[idx:idx+2])

        # Compute angles of the start and target lines (angle of normal direction)
        angles_start[i] = line_angle(line_start)
        angles_target[i] = line_angle(line_target)

        # Ensure angles are within [0, 2π)
        angles_start[i] = mod(angles_start[i], 2π)
        angles_target[i] = mod(angles_target[i], 2π)
        # Ensure the difference between start and target angles is within [-π, π)
        angles_diff[i] = angles_target[i] - angles_start[i]
        if angles_diff[i] > π
            angles_diff[i] -= 2π
        elseif angles_diff[i] < -π
            angles_diff[i] += 2π
        end
    end

    InfiniteHomographyHomotopy(
        F,
        p̂,
        q̂,
        H_infs_f,
        log_H_infs,
        vps_per_group,
        Vector{Int}(h_indices),
        Vector{Float64}(angles_start),
        Vector{Float64}(angles_target),
        Vector{Float64}(angles_diff),
        Ref(complex(NaN)),
        pt,
        taylor_pt,
        H_t_cache
    )
end

Base.size(H::InfiniteHomographyHomotopy) = size(H.F)

function start_parameters!(H::InfiniteHomographyHomotopy, p)
    H.p .= p
    H.t_cache[] = NaN
    H
end

function target_parameters!(H::InfiniteHomographyHomotopy, q)
    H.q .= q
    H.t_cache[] = NaN
    H
end

function parameters!(H::InfiniteHomographyHomotopy, p, q)
    H.p .= p
    H.q .= q
    H.t_cache[] = NaN
    H
end

"""
    interpolate_line(H, line_idx, t)

Interpolate line `line_idx` at parameter t ∈ [0,1].
Returns the interpolated line in homogeneous coordinates.

Note: HomotopyContinuation convention is t=1 → start, t=0 → target.

Uses intersection-point interpolation: the line passes through its vanishing point
and a linearly interpolated intersection point, avoiding parallel-line singularities.
"""
function interpolate_line(H::InfiniteHomographyHomotopy, line_idx::Int, t::Real)
    g = H.h_indices[line_idx]  # Get group for this line

    # Compute H_t: at t=1 → I, at t=0 → H_∞
    H_t = matrix_exp((1 - t) * H.log_H_infs[g])

    v_0 = H.vanishing_points_per_group[g]  # Get vanishing point for this group
    v_t = H_t * v_0  # Transform vanishing point with H_t (homogeneous 3D)

    # Compute start and target intersection points (using first two lines)
    start_line1 = real.(H.p[1:3])
    start_line2 = real.(H.p[4:6])
    start_intersection = cross(start_line1, start_line2)
    start_intersection = start_intersection / start_intersection[3]

    target_line1 = real.(H.q[1:3])
    target_line2 = real.(H.q[4:6])
    target_intersection = cross(target_line1, target_line2)
    target_intersection = target_intersection / target_intersection[3]

    # Interpolate intersection point: t=1 → start, t=0 → target
    intersection_t = t * start_intersection + (1 - t) * target_intersection

    # Line through vanishing point and intersection point
    line_t = line_through_two_points(v_t, intersection_t)

    return line_t
end

"""
    line_through_two_points(p1, p2)

Compute the line passing through two homogeneous points.
Returns the line in homogeneous coordinates [a, b, c] such that a*x + b*y + c = 0.
"""
function line_through_two_points(p1::AbstractVector, p2::AbstractVector)
    line = cross(p1, p2)
    return line / norm(line)
end

"""
    compute_parameters_at_t(H, t)

Compute the interpolated line parameters at time t (without caching).
Returns a vector of length 3*number_of_lines.

Note: HomotopyContinuation convention is t=1 → start, t=0 → target.
So we interpolate: params(t) = params_start when t=1, params_target when t=0.

Algorithm: interpolate the intersection point of all lines, then reconstruct
each line as passing through its vanishing point and the interpolated intersection.
This avoids singularities where lines become parallel.
"""
function compute_parameters_at_t(H::InfiniteHomographyHomotopy, t::Real)
    number_of_lines = length(H.h_indices)
    parameters = zeros(3 * number_of_lines)

    # First, compute start and target intersection points
    # Start intersection: cross product of first two start lines (t=1)
    start_line1 = real.(H.p[1:3])
    start_line2 = real.(H.p[4:6])
    start_intersection = cross(start_line1, start_line2)
    start_intersection = start_intersection / start_intersection[3]  # Dehomogenize

    # Target intersection: cross product of first two target lines (t=0)
    target_line1 = real.(H.q[1:3])
    target_line2 = real.(H.q[4:6])
    target_intersection = cross(target_line1, target_line2)
    target_intersection = target_intersection / target_intersection[3]  # Dehomogenize

    # Interpolate intersection point: t=1 → start, t=0 → target
    # intersection(t) = t * start + (1-t) * target
    intersection_t = t * start_intersection + (1 - t) * target_intersection

    for i in 1:number_of_lines
        idx = (i-1)*3 + 1
        g = H.h_indices[i]

        # Compute H_t and vanishing point at time t
        # At t=1: H_t = I, v_t = v_0 (start)
        # At t=0: H_t = H_∞, v_t = H_∞ * v_0 (target)
        H_t = matrix_exp((1 - t) * H.log_H_infs[g])
        v_0 = H.vanishing_points_per_group[g]
        v_t = H_t * v_0

        # Line through vanishing point and intersection point
        line_t = line_through_two_points(v_t, intersection_t)

        parameters[idx:idx+2] = line_t
    end

    return parameters
end

"""
    tp!(H, t)

Compute interpolated parameters at time t and update the Taylor vector cache.
Uses numerical differentiation for the Taylor coefficients since the interpolation is non-linear.
"""
function tp!(H::InfiniteHomographyHomotopy, tinput::Union{ComplexF64,Float64})
    tinput == H.t_cache[] && return H.taylor_pt
    t = real(tinput)

    # Compute parameters at current t
    parameters = compute_parameters_at_t(H, t)

    # Compute derivative numerically using central differences
    ε = 1e-7
    t_lo = max(0.0, t - ε)
    t_hi = min(1.0, t + ε)
    Δt = t_hi - t_lo

    params_lo = compute_parameters_at_t(H, t_lo)
    params_hi = compute_parameters_at_t(H, t_hi)
    derivatives = (params_hi - params_lo) / Δt

    @inbounds for i = 1:length(H.taylor_pt)
        ptᵢ = parameters[i]
        H.pt[i] = ptᵢ
        H.taylor_pt[i] = (ptᵢ, derivatives[i])
    end
    H.t_cache[] = tinput

    H.taylor_pt
end

function ModelKit.evaluate!(u, H::InfiniteHomographyHomotopy, x, t)
    tp!(H, t)
    evaluate!(u, H.F, x, H.pt)
end

function ModelKit.evaluate_and_jacobian!(u, U, H::InfiniteHomographyHomotopy, x, t)
    tp!(H, t)
    evaluate_and_jacobian!(u, U, H.F, x, H.pt)
end

function ModelKit.taylor!(u, v::Val, H::InfiniteHomographyHomotopy, tx, t)
    taylor!(u, v, H.F, tx, tp!(H, t))
    u
end

"""
    verify_line_through_vanishing_point(H, line_idx, t)

Verify that the interpolated line at time t passes through the interpolated vanishing point.
Returns the absolute value of ℓ_tᵀ * v_t (should be ≈ 0).

Note: Uses HC convention where t=1 → start, t=0 → target.
"""
function verify_line_through_vanishing_point(H::InfiniteHomographyHomotopy, line_idx::Int, t::Real)
    g = H.h_indices[line_idx]  # Get group for this line

    # Compute v_t = H_t * v₀ using THIS GROUP's H_inf
    # HC convention: t=1 → I (start), t=0 → H_∞ (target)
    H_t = matrix_exp((1 - t) * H.log_H_infs[g])
    v_t = H_t * H.vanishing_points_per_group[g]
    v_t = normalize(v_t)

    # Get interpolated line
    line_t = interpolate_line(H, line_idx, t)

    # Check incidence: ℓ_tᵀ * v_t should be 0
    return abs(dot(line_t, v_t))
end

"""
    get_vanishing_point(H, line_idx)

Return the vanishing point (at t=0) associated with line `line_idx`.
"""
function get_vanishing_point(H::InfiniteHomographyHomotopy, line_idx::Int)
    g = H.h_indices[line_idx]
    return H.vanishing_points_per_group[g]
end
