#!/usr/bin/env julia
"""
Generate synthetic scene data for the Ceres solver comparison.

Outputs:
  ceres_solver/scene_data.json        (no noise)
  ceres_solver/scene_data_noisy.json  (with --noise <pixels>)

Usage:
  julia scripts/generate_ceres_scene.jl
  julia scripts/generate_ceres_scene.jl --noise 1.0
"""

ENV["GUI_ENABLED"] = "false"
using Pkg
Pkg.activate(dirname(dirname(abspath(@__FILE__))))

using CylindersBasedCameraResectioning
using LinearAlgebra
using Random
using JSON
using Rotations

const CylM   = CylindersBasedCameraResectioning.Cylinder
const CamM   = CylindersBasedCameraResectioning.Camera
const GeomM  = CylindersBasedCameraResectioning.Geometry
const UtilM  = CylindersBasedCameraResectioning.Utils
const SceneM = CylindersBasedCameraResectioning.Scene

# ─── CLI arguments ────────────────────────────────────────────────────────────

noise = let
    val = 0.0
    for (i, arg) in enumerate(ARGS)
        if arg == "--noise" && i < length(ARGS)
            val = parse(Float64, ARGS[i + 1])
        end
    end
    val
end

# ─── Scene generation ────────────────────────────────────────────────────────

Random.seed!(777)

# 3-cylinder calibration rig (arbitrary_rig seeds itself at 2300 internally,
# matching the rig used in Lab.compare_parameter_homotopies)
cylinders = CylM.CalibrationRigs.arbitrary_rig()

# Common intrinsics shared by both cameras (zero skew)
K_true = [
    2600.0  0.0   960.0;
    0.0  1500.0   540.0;
    0.0     0.0     1.0
]

# Two cameras at random positions looking at the origin
cameras = map(1:2) do _
    pos, rot = CamM.random_camera_lookingat_center()
    cam = CamM.CameraProperties()
    cam.position         = pos
    cam.quaternion_rotation = rot
    cam.intrinsic        = K_true
    cam
end

# Silhouette lines for each camera  – shape: (n_cylinders, 2, 3)
views = [GeomM.get_view(cylinders, cam) for cam in cameras]

# Stack to (2*n_cylinders, 3) so row k matches points_at_infinity row k
stacked_views = [UtilM.lines_clp_to_stack(v) for v in views]

# Cylinder axes in world space – shape (2*n_cylinders, 3)
pts_inf, _ = CylM.points_at_infinity_dualquadrics(cylinders)

# ─── Verify noiseless equations ───────────────────────────────────────────────

println("Verifying  l' * K * R * d = 0  for true parameters (before noise):")
for (vi, lines) in enumerate(stacked_views)
    R = Float64.(collect(cameras[vi].rotation_matrix))
    for li in axes(lines, 1)
        l = lines[li, :]
        d = pts_inf[li, :]
        res = dot(l, K_true * R * d)
        @assert abs(res) < 1e-6 "View $vi line $li: |residual| = $(abs(res))"
    end
    println("  View $vi: $(size(lines, 1)) residuals < 1e-6 ✓")
end

# ─── Apply noise to line observations ────────────────────────────────────────

if noise > 0.0
    println("\nApplying noise = $noise pixels to line observations:")
    for (vi, lines) in enumerate(stacked_views)
        for li in axes(lines, 1)
            noisy = SceneM.add_noise_to_line(lines[li, :], noise)
            lines[li, :] = normalize(noisy)
        end
        R = Float64.(collect(cameras[vi].rotation_matrix))
        residuals_after = [abs(dot(lines[li, :], K_true * R * pts_inf[li, :]))
                           for li in axes(lines, 1)]
        println("  View $vi: max |residual| after noise = $(round(maximum(residuals_after), sigdigits=3))")
    end
end

# ─── Perturbed starting solution ─────────────────────────────────────────────

gamma = 23.0    # position-offset magnitude (matching compare_parameter_homotopies)
beta  = gamma # 23.0    # additional rotation offset in degrees

perturbed_K = K_true + [
    (rand() * 50.0 - 25.0)  0.0  (rand() * 10.0 - 5.0);
    0.0  (rand() * 50.0 - 25.0)  (rand() * 10.0 - 5.0);
    0.0   0.0   0.0
]

perturbed_cameras = map(cameras) do cam
    pcam = CamM.CameraProperties()
    pcam.intrinsic = perturbed_K
    # Move camera position in a random direction
    dir = normalize(randn(3))
    pcam.position = cam.position + gamma * dir
    # Recompute lookat rotation from the new position
    pcam.quaternion_rotation = CamM.lookat_quaternion(pcam.position, [0.0, 0.0, 0.0])
    # Apply an additional small rotation of beta degrees around a random axis
    rand_axis = normalize(randn(3))
    extra = QuatRotation(AngleAxis(deg2rad(beta), rand_axis...))
    pcam.quaternion_rotation = extra * pcam.quaternion_rotation
    pcam
end

# Silhouette lines for perturbed cameras – used as start lines for Julia homotopy
start_views = [GeomM.get_view(cylinders, pcam) for pcam in perturbed_cameras]
start_stacked_views = [UtilM.lines_clp_to_stack(v) for v in start_views]

# ─── Helper: rotation matrix → angle-axis vector ─────────────────────────────

function rot_to_aa(R_mat::Matrix{Float64})
    cos_θ = clamp((R_mat[1,1] + R_mat[2,2] + R_mat[3,3] - 1.0) / 2.0, -1.0, 1.0)
    θ = acos(cos_θ)
    if θ < 1e-10
        return [0.0, 0.0, 0.0]
    end
    ax = [R_mat[3,2] - R_mat[2,3],
          R_mat[1,3] - R_mat[3,1],
          R_mat[2,1] - R_mat[1,2]]
    ax_norm = norm(ax)
    if ax_norm < 1e-10
        # θ ≈ π – recover axis from (R + I)
        M = R_mat + I(3)
        col_norms = [norm(M[:, j]) for j in 1:3]
        ax = normalize(M[:, argmax(col_norms)])
    else
        ax = ax / ax_norm
    end
    return θ .* ax
end

# ─── Build and write JSON ─────────────────────────────────────────────────────

residuals_json = []
for (vi, lines) in enumerate(stacked_views)
    start_lines = start_stacked_views[vi]
    for li in axes(lines, 1)
        push!(residuals_json, Dict(
            "view" => vi - 1,          # 0-indexed for C++
            "line" => lines[li, :],
            "axis" => pts_inf[li, :],
            "start_line" => start_lines[li, :]
        ))
    end
end

scene_data = Dict(
    "n_views"     => length(cameras),
    "n_cylinders" => length(cylinders),
    "residuals"   => residuals_json,
    "true" => Dict(
        "intrinsics" => Dict(
            "fx" => K_true[1,1], "fy" => K_true[2,2],
            "cx" => K_true[1,3], "cy" => K_true[2,3]
        ),
        "rotations" => [
            Dict("angle_axis" => rot_to_aa(Float64.(collect(cam.rotation_matrix))))
            for cam in cameras
        ]
    ),
    "start" => Dict(
        "intrinsics" => Dict(
            "fx" => perturbed_K[1,1], "fy" => perturbed_K[2,2],
            "cx" => perturbed_K[1,3], "cy" => perturbed_K[2,3]
        ),
        "rotations" => [
            Dict("angle_axis" => rot_to_aa(Float64.(collect(pcam.rotation_matrix))))
            for pcam in perturbed_cameras
        ]
    )
)

out_dir = joinpath(dirname(dirname(abspath(@__FILE__))), "ceres_solver")
isdir(out_dir) || mkdir(out_dir)
out_filename = noise > 0.0 ? "scene_data_noisy.json" : "scene_data.json"
out_path = joinpath(out_dir, out_filename)
open(out_path, "w") do f
    JSON.print(f, scene_data, 2)
end

# ─── Summary ─────────────────────────────────────────────────────────────────

println("\nScene data written to $out_path")
println("\nPerturbation parameters:")
println("  gamma (position offset) = $gamma")
println("  beta  (rotation offset) = $(beta)°")
println("  noise (line obs, pixels) = $noise")
println("\nIntrinsic offsets (start − true):")
println("  Δfx = $(round(perturbed_K[1,1]-K_true[1,1], digits=2))")
println("  Δfy = $(round(perturbed_K[2,2]-K_true[2,2], digits=2))")
println("  Δcx = $(round(perturbed_K[1,3]-K_true[1,3], digits=2))")
println("  Δcy = $(round(perturbed_K[2,3]-K_true[2,3], digits=2))")
println("\nRotation offset (start vs true):")
for (vi, (cam, pcam)) in enumerate(zip(cameras, perturbed_cameras))
    R1 = Float64.(collect(cam.rotation_matrix))
    R2 = Float64.(collect(pcam.rotation_matrix))
    cos_θ = clamp((sum(R1 .* R2) - 1.0) / 2.0, -1.0, 1.0)
    println("  View $vi: $(round(acosd(cos_θ), digits=2))°")
end
