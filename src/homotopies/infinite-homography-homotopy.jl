export InfiniteHomographyHomotopy, interpolate_line, verify_line_through_vanishing_point, matrix_exp, matrix_log, tp!, get_vanishing_point

using LinearAlgebra: cross, norm, normalize, dot, pinv, I, eigen, Diagonal
using HomotopyContinuation

"""
    line_angle(line1, line2)

Compute the angle between two lines in homogeneous coordinates. Second line defaults to X-axis line [1, 0, 0].
Returns the angle in radians.
"""
function line_angle(line1::AbstractVector, line2::AbstractVector = [1.0, 0.0, 0.0])
    cos_theta = abs(dot(line1, line2)) / (norm(line1) * norm(line2))
    return acos(clamp(cos_theta, -1.0, 1.0))
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
the homotopy interpolates lines as follow:
1. Compute angle of each starting and target line w.r.t the X-axis line [1, 0, 0]
2. Computes angle difference difference between the starting and target lines
3. Interpolate vanishing point: v_t = H_t * v₀
4. Interpolate line angle: θ_t = (1-t)*θ₀ + t*θ₁
5. Reconstruct ℓ_t as line passing by v_t with angle θ_t: ℓ_t = [cos(θ_t), sin(θ_t), -dot(v_t, [cos(θ_t), sin(θ_t)])]

This ensures ℓ_t always passes through v_t = H_t * v₀.
"""
struct InfiniteHomographyHomotopy{T<:AbstractSystem} <: AbstractHomotopy
    F::T
    p::Vector{ComplexF64}  # start parameters (flattened lines)
    q::Vector{ComplexF64}  # target parameters (flattened lines)

    # Arrays of H_inf matrices (one per group)
    H_infs::Vector{Matrix{Float64}}        # H_∞ for each group
    H_inf_invs::Vector{Matrix{Float64}}    # H_∞⁻¹ for each group
    H_inf_invTs::Vector{Matrix{Float64}}   # H_∞^{-T} for each group
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
    H_t_invT_cache::Vector{Matrix{Float64}}
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
    H_inf_invs = [Matrix{Float64}(inv(H)) for H in H_infs_f]
    H_inf_invTs = [Matrix{Float64}(inv(H)') for H in H_infs_f]
    log_H_infs = [matrix_log(H) for H in H_infs_f]

    # Normalize vanishing points per group
    vps_per_group = [normalize(Vector{Float64}(v)) for v in vanishing_points_per_group]

    # Initialize H_t caches (will be computed in tp!)
    H_t_cache = [zeros(Float64, 3, 3) for _ in 1:num_groups]
    H_t_invT_cache = [zeros(Float64, 3, 3) for _ in 1:num_groups]

    # Precompute per-line data
    angles_start = zeros(Float64, number_of_lines)
    angles_target = zeros(Float64, number_of_lines)
    angles_diff = zeros(Float64, number_of_lines)

    for i in 1:number_of_lines
        idx = (i-1)*3 + 1
        line_start = real.(p[idx:idx+2])
        line_target = real.(q[idx:idx+2])

        # Get the group index and corresponding vanishing point
        g = h_indices[i]
        v0 = vps_per_group[g]

        # Compute angles of the start and target lines with respect to the vanishing point
        angles_start[i] = line_angle(line_start, v0)
        angles_target[i] = line_angle(line_target, v0)

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
        H_inf_invs,
        H_inf_invTs,
        log_H_infs,
        vps_per_group,
        Vector{Int}(h_indices),
        Vector{Float64}(angles_start),
        Vector{Float64}(angles_target),
        Vector{Float64}(angles_diff),
        Ref(complex(NaN)),
        pt,
        taylor_pt,
        H_t_cache,
        H_t_invT_cache
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
"""
function interpolate_line(H::InfiniteHomographyHomotopy, line_idx::Int, t::Real)
    g = H.h_indices[line_idx]  # Get group for this line

    # Compute H_t = exp(t * log(H_∞)) for this group
    H_t = matrix_exp(t * H.log_H_infs[g])
    H_t_invT = inv(H_t)'

    # Interpolate angle for this line
    θ_t = H.angles_start[line_idx] + t * H.angles_diff[line_idx]
    
    v_0 = H.vps_per_group[g]  # Get vanishing point for this group
    v_t = H_t * v_0  # Transform vanishing point with H_t

    # Reconstruct interpolated line
    line_t = [cos(θ_t), sin(θ_t), -dot(v_t, [cos(θ_t), sin(θ_t)])]

    return normalize(line_t)
end

"""
    tp!(H, t)

Compute interpolated parameters at time t and update the Taylor vector cache.
"""
function tp!(H::InfiniteHomographyHomotopy, tinput::Union{ComplexF64,Float64})
    tinput == H.t_cache[] && return H.taylor_pt
    t = real(tinput)

    number_of_lines = length(H.h_indices)
    num_groups = length(H.H_infs)
    parameters = zeros(3 * number_of_lines)

    # Compute H_t for each group ONCE (optimization)
    for g in 1:num_groups
        H_t = matrix_exp(t * H.log_H_infs[g])
        H.H_t_cache[g] .= H_t
        H.H_t_invT_cache[g] .= inv(H_t)'
    end

    for i in 1:number_of_lines
        idx = (i-1)*3 + 1
        g = H.h_indices[i]  # Get group for this line

        # Compute vanishing point in homogeneous coordinates
        v_0 = H.vps_per_group[g]
        v_t = H.H_t_cache[g] * v_0

        # Interpolate pencil coordinates
        θ_t = H.angles_start[i] + t * H.angles_diff[i]

        # Reconstruct interpolated line
        line_t = [cos(θ_t), sin(θ_t), -dot(v_t, [cos(θ_t), sin(θ_t)])]

        # Normalize for numerical stability
        line_t = line_t / norm(line_t)

        parameters[idx:idx+2] = line_t
    end

    @inbounds for i = 1:length(H.taylor_pt)
        ptᵢ = parameters[i]
        H.pt[i] = ptᵢ
        H.taylor_pt[i] = (ptᵢ, H.p[i] - H.q[i])
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
"""
function verify_line_through_vanishing_point(H::InfiniteHomographyHomotopy, line_idx::Int, t::Real)
    g = H.h_indices[line_idx]  # Get group for this line

    # Compute v_t = H_t * v₀ using THIS GROUP's H_inf
    H_t = matrix_exp(t * H.log_H_infs[g])
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
