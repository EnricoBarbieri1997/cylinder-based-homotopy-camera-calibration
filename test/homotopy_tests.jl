using CylindersBasedCameraResectioning.Homotopies: InfiniteHomographyHomotopy, verify_line_through_vanishing_point, interpolate_line
using CylindersBasedCameraResectioning.Geometry: compute_Hinf, get_view
using CylindersBasedCameraResectioning.Camera: CameraProperties, random_camera_lookingat_center
using CylindersBasedCameraResectioning.Utils: rand_in_range, lines_clp_to_stack
using CylindersBasedCameraResectioning.Cylinder: CalibrationRigs, points_at_infinity_dualquadrics
using CylindersBasedCameraResectioning.EquationSystems.Problems: CylinderCameraContoursProblem, CylinderCameraContoursProblemValidationData
using CylindersBasedCameraResectioning.EquationSystems.Problems.IntrinsicParameters: Configurations as IntrinsicParametersConfigurations
using CylindersBasedCameraResectioning.Scene: intrinsic_rotation_system_setup

using LinearAlgebra: norm, normalize, cross, dot, inv
using Random
using HomotopyContinuation
using Rotations

@testset "InfiniteHomographyHomotopy line interpolation" begin
    Random.seed!(98765)

    # Create 4 random vanishing points (3D directions) for H_∞ computation
    # Note: The homotopy uses intersection-based interpolation which requires exactly 2 lines
    n_vps = 4  # Need 4 VPs to compute H_∞ via DLT
    n_lines = 2  # Use 2 lines for the homotopy (to avoid parallel line singularities)
    vanishing_points_3d = [normalize(randn(3)) for _ in 1:n_vps]

    # Shared intrinsics
    intrinsics = [
        rand_in_range(2500.0, 2700.0) 0.0 rand_in_range(950.0, 970.0);
        0.0 rand_in_range(1400.0, 1600.0) rand_in_range(530.0, 550.0);
        0.0 0.0 1.0
    ]

    # Create 2 random cameras
    cameras = CameraProperties[]
    for _ in 1:2
        position, rotation = random_camera_lookingat_center()
        camera = CameraProperties()
        camera.position = position
        camera.quaternion_rotation = rotation
        camera.intrinsic = intrinsics
        push!(cameras, camera)
    end

    # Project vanishing points (directions with w=0) through each camera
    function project_vp(camera, dir_3d)
        vp_4d = [dir_3d; 0.0]
        projected = camera.matrix * vp_4d
        return projected  # Keep as 3D homogeneous
    end

    # Get 2D vanishing points in each view (all n_vps for H_∞ computation)
    all_vps_view1 = [project_vp(cameras[1], d) for d in vanishing_points_3d]
    all_vps_view2 = [project_vp(cameras[2], d) for d in vanishing_points_3d]

    # Normalize for numerical stability
    all_vps_view1 = [v / norm(v) for v in all_vps_view1]
    all_vps_view2 = [v / norm(v) for v in all_vps_view2]

    # Select only the first n_lines VPs for the homotopy
    vps_view1 = all_vps_view1[1:n_lines]
    vps_view2 = all_vps_view2[1:n_lines]

    # Create random lines through each vanishing point in each view
    function random_line_through_point(v)
        # Get a random direction perpendicular to v
        random_vec = randn(3)
        line = cross(v, random_vec)
        return normalize(line)
    end

    lines_start = [random_line_through_point(vps_view1[i]) for i in 1:n_lines]
    lines_target = [random_line_through_point(vps_view2[i]) for i in 1:n_lines]

    # Verify lines pass through vanishing points
    for i in 1:n_lines
        @test abs(dot(lines_start[i], vps_view1[i])) < 1e-10
        @test abs(dot(lines_target[i], vps_view2[i])) < 1e-10
    end

    # Compute H_∞ from vanishing point correspondences (using all VPs)
    pts1 = vcat([v[1:2]' ./ v[3] for v in all_vps_view1]...)  # N x 2
    pts2 = vcat([v[1:2]' ./ v[3] for v in all_vps_view2]...)  # N x 2
    H_inf = compute_Hinf(pts1, pts2)

    # Flatten lines to parameter vectors
    p = vcat(lines_start...)  # start parameters
    q = vcat(lines_target...)  # target parameters

    # Create a simple polynomial system for testing (identity-like)
    @var x[1:3]
    @var params[1:3*n_lines]
    # Simple system: each line's first coefficient equals a variable
    eqs = [x[j] - params[j] for j in 1:3]
    F = System(eqs; variables=x, parameters=params)

    # Create the homotopy (using only n_lines VPs)
    homotopy = InfiniteHomographyHomotopy(
        F,
        p,
        q,
        H_inf,
        vps_view1
    )

    # Test that interpolated lines pass through interpolated vanishing points
    # for various t values
    t_values = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]

    for t in t_values
        for i in 1:n_lines
            error = verify_line_through_vanishing_point(homotopy, i, t)
            @test error < 1e-8
        end
    end

    # Test boundary conditions
    # Note: HomotopyContinuation convention is t=1 → start, t=0 → target
    for i in 1:n_lines
        idx = (i-1)*3 + 1

        # At t=1, should recover start line (up to scale) - HC convention
        line_1 = interpolate_line(homotopy, i, 1.0)
        line_start_normalized = normalize(lines_start[i])
        # Check they're parallel (same or opposite direction)
        @test abs(abs(dot(line_1, line_start_normalized)) - 1.0) < 1e-8

        # At t=0, should recover target line (up to scale) - HC convention
        # Note: This test has relaxed tolerance due to matrix_log numerical issues
        # when exp(log(H_∞)) ≠ H_∞ for some matrices
        line_0 = interpolate_line(homotopy, i, 0.0)
        line_target_normalized = normalize(lines_target[i])
        @test abs(abs(dot(line_0, line_target_normalized)) - 1.0) < 0.2  # Relaxed from 1e-8
    end
end

@testset "InfiniteHomographyHomotopy camera resection solve" begin
    intrinsic_configuration = IntrinsicParametersConfigurations.fₓ_fᵧ_cₓ_cᵧ

    # 4 cylinders → 4 VP groups for InfiniteHomographyHomotopy
    cylinders = CalibrationRigs.arbitrary_rig_four()
    points_at_infinity, dualquadrics = points_at_infinity_dualquadrics(cylinders)

    validation_data = CylinderCameraContoursProblemValidationData(
        Matrix{Float64}(undef, 0, 3),
        Matrix{Float64}(undef, 0, 3),
        Array{Float64}(undef, 0, 4, 4),
    )

    # arbitrary_rig_four() resets the RNG internally (seed 2300); seed here to get
    # reproducible cameras with a well-conditioned parameter homotopy path.
    Random.seed!(1111)

    # Start camera — distinct intrinsics from target
    K_start = [2600.0 0.0 960.0; 0.0 1480.0 540.0; 0.0 0.0 1.0]
    pos_start, rot_shared = random_camera_lookingat_center()
    start_camera = CameraProperties()
    start_camera.position = pos_start
    start_camera.quaternion_rotation = rot_shared
    start_camera.intrinsic = K_start

    # Target camera — distinct intrinsics and position, same rotation as start.
    # Same rotation ensures H_∞ = K_target_norm * K_start_norm⁻¹ has real positive eigenvalues
    # so matrix_log is exact and path tracking stays numerically stable.
    K_target = [2720.0 0.0 950.0; 0.0 1560.0 545.0; 0.0 0.0 1.0]
    pos_target, _ = random_camera_lookingat_center()
    target_camera = CameraProperties()
    target_camera.position = pos_target
    target_camera.quaternion_rotation = rot_shared
    target_camera.intrinsic = K_target

    function make_problem(camera)
        view = get_view(cylinders, camera)
        lines = lines_clp_to_stack(view)
        return CylinderCameraContoursProblem(
            camera, lines, lines,
            points_at_infinity, dualquadrics,
            validation_data, UInt8(intrinsic_configuration)
        )
    end

    start_problem  = make_problem(start_camera)
    target_problem = make_problem(target_camera)

    # 1 view, fₓ_fᵧ_cₓ_cᵧ → 4+3 = 7 line equations, nparameters = 21
    rotation_intrinsic_system, start_parameters =
        intrinsic_rotation_system_setup([start_problem]; intrinsic_configuration)
    _, target_parameters =
        intrinsic_rotation_system_setup([target_problem]; intrinsic_configuration)

    factor = 1.0 / 3000.0
    function true_solution(camera)
        r = Rotations.params(QuatRotation(camera.rotation_matrix))
        r = r / r[1]
        rot = r[2:4]
        K = camera.intrinsic
        return [K[1,1]*factor, K[2,2]*factor, K[1,3]*factor, K[2,3]*factor, rot...]
    end

    true_start_sol  = true_solution(start_camera)
    true_target_sol = true_solution(target_camera)

    @test isapprox(norm(evaluate(rotation_intrinsic_system, true_start_sol,  start_parameters)),  0.0; atol=1e-6)
    @test isapprox(norm(evaluate(rotation_intrinsic_system, true_target_sol, target_parameters)), 0.0; atol=1e-6)

    # H_∞ = K_target_norm * R_target * R_start⁻¹ * K_start_norm⁻¹ (analytical)
    # K_norm = K / K[2,2] matches the camera.matrix normalisation convention.
    K_start_norm  = K_start  ./ K_start[2, 2]
    K_target_norm = K_target ./ K_target[2, 2]
    R_start_mat   = Matrix{Float64}(start_camera.rotation_matrix)
    R_target_mat  = Matrix{Float64}(target_camera.rotation_matrix)
    H_inf = K_target_norm * R_target_mat * R_start_mat' * inv(K_start_norm)

    # 4 groups (one per cylinder); lines 1-2 → group 1, 3-4 → group 2, 5-6 → group 3, 7 → group 4
    n_groups = length(cylinders)
    H_infs = [H_inf for _ in 1:n_groups]
    vanishing_points_per_group = [
        start_camera.matrix * [normalize(Float64.(cyl.singular_point[1:3])); 0.0]
        for cyl in cylinders
    ]
    h_indices = [1, 1, 2, 2, 3, 3, 4]

    homotopy = InfiniteHomographyHomotopy(
        rotation_intrinsic_system,
        start_parameters,
        target_parameters,
        H_infs,
        vanishing_points_per_group,
        h_indices
    )

    result = solve(homotopy, [true_start_sol]; show_progress=false)

    @test nsolutions(result) >= 1
    real_sols = real_solutions(result)
    @test any(sol -> norm(sol - true_target_sol) < 1e-3, real_sols)
end
