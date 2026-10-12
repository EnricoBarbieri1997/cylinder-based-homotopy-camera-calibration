module Visualizations
    export run_four_homotopies_tracking,
           visualize_four_homotopies,
           setup_toy_problem,
           track_with_steps,
           solution_to_pose,
           default_tracked_path

    using HomotopyContinuation
    using LinearAlgebra: norm, normalize, dot, cross, det, I
    using Rotations
    using Random
    using Serialization

    using ..Camera: CameraProperties, random_camera_lookingat_center
    using ..Utils: rand_in_range
    using ..Geometry: compute_Hinf
    using ..Homotopies: InfiniteHomographyHomotopy, GeometricHomotopy
    using ..Homotopies: ParameterHomotopy as MyParameterHomotopy
    using ..CylindersBasedCameraResectioning: GUI_ENABLED

    const default_tracked_path = "./tmp/four_homotopies_tracked.jls"

    const HOMOTOPY_ORDER = ("InfiniteHomography", "Geometric", "Parameter", "TotalDegree")
    const HOMOTOPY_COLORS = Dict(
        "InfiniteHomography" => :red,
        "Geometric"          => :green,
        "Parameter"          => :blue,
        "TotalDegree"        => :purple,
    )

    """
        quaternion_rot(w, x, y, z)

    |q|² · R for q = (w, x, y, z) — same convention as `infinite_homography_resection_homotopy`.
    """
    function quaternion_rot(w::Number, x::Number, y::Number, z::Number)
        return [
            w^2+x^2-y^2-z^2  2*(x*y-w*z)      2*(x*z+w*y);
            2*(x*y+w*z)      w^2-x^2+y^2-z^2  2*(y*z-w*x);
            2*(x*z-w*y)      2*(y*z+w*x)      w^2-x^2-y^2+z^2
        ]
    end

    """
        solution_to_pose(sol, R₁)

    Decode a solution vector [a, b, c, τ₁, τ₂, τ₃] back into (R, t), using the
    same chart (R = R_rel · R₁ with w = 1 quaternion) as the toy problem.
    """
    function solution_to_pose(sol::AbstractVector{<:Real}, R₁::AbstractMatrix{<:Real})
        s = 1 + sol[1]^2 + sol[2]^2 + sol[3]^2
        R = quaternion_rot(1, sol[1], sol[2], sol[3]) * R₁ ./ s
        t = Vector{Float64}(sol[4:6]) ./ s
        return R, t
    end

    """
        setup_toy_problem(; random_seed=84564)

    Reproduce the identical toy scene / polynomial system / start-target solutions
    used by `Lab.infinite_homography_resection_homotopy(random_seed=84564)`.
    Returns a NamedTuple holding everything the four homotopies need, plus the
    ground-truth cameras for the visualization.
    """
    function setup_toy_problem(; random_seed=84564)
        Random.seed!(random_seed)

        n_lines = 4
        assignment = [(1, 2, 1), (1, 4, 1), (2, 3, 2), (1, 3, 3), (3, 4, 3), (2, 4, 4)]

        vanishing_points_3d = [normalize(randn(3)) for _ in 1:n_lines]
        points_3d = [randn(3) for _ in 1:n_lines]

        intrinsics = [
            rand_in_range(2500.0, 2700.0) 0.0 rand_in_range(950.0, 970.0);
            0.0 rand_in_range(1400.0, 1600.0) rand_in_range(530.0, 550.0);
            0.0 0.0 1.0
        ]

        cameras = CameraProperties[]
        for _ in 1:2
            position, rotation = random_camera_lookingat_center()
            camera = CameraProperties()
            camera.position = position
            camera.quaternion_rotation = rotation
            camera.intrinsic = intrinsics
            push!(cameras, camera)
        end

        K = intrinsics ./ intrinsics[2, 2]
        rotations_gt = [Matrix{Float64}(camera.rotation_matrix) for camera in cameras]
        translations_gt = [-rotations_gt[i] * Vector{Float64}(cameras[i].position) for i in 1:2]
        calibrated_matrices = [K \ camera.matrix for camera in cameras]

        vps_views = [[normalize(P * [v; 0.0]) for v in vanishing_points_3d] for P in calibrated_matrices]
        points_views = [[P * [w; 1.0] for w in points_3d] for P in calibrated_matrices]
        lines_views = [[normalize(cross(vps_views[k][i], points_views[k][i])) for i in 1:n_lines] for k in 1:2]

        pts1 = vcat([v[1:2]' ./ v[3] for v in vps_views[1]]...)
        pts2 = vcat([v[1:2]' ./ v[3] for v in vps_views[2]]...)
        Hinf = compute_Hinf(pts1, pts2)
        Hinf = sign(det(Hinf)) * Hinf

        @var a b c τ[1:3]
        @var l[1:3, 1:n_lines]
        R₁ = rotations_gt[1]
        R_q = quaternion_rot(1, a, b, c)
        projected_lines = [cross(R_q * R₁ * vanishing_points_3d[i], R_q * R₁ * points_3d[i] + τ) for i in 1:n_lines]
        line_params = [l[:, k] for k in 1:n_lines]
        equations = [
            sum(projected_lines[owner] .* cross(line_params[i], line_params[k]))
            for (i, k, owner) in assignment
        ]
        F = System(equations; variables=[a, b, c, τ...], parameters=vec(l))

        p = vcat(lines_views[1]...)
        q = vcat(lines_views[2]...)

        pose_to_solution = function (R, t)
            qr = QuatRotation(R * R₁')
            w, x, y, z = qr.q.s, qr.q.v1, qr.q.v2, qr.q.v3
            a₀, b₀, c₀ = x / w, y / w, z / w
            s = 1 + a₀^2 + b₀^2 + c₀^2
            return [a₀, b₀, c₀, (s * t)...]
        end

        start_sol = pose_to_solution(rotations_gt[1], translations_gt[1])
        target_sol = pose_to_solution(rotations_gt[2], translations_gt[2])

        camera_positions_world = [Vector{Float64}(c.position) for c in cameras]

        return (;
            F=F, p=p, q=q, Hinf=Hinf,
            vps_view1=vps_views[1], vps_view2=vps_views[2],
            vanishing_points_3d=vanishing_points_3d,
            points_3d=points_3d,
            intrinsics=intrinsics,
            cameras=cameras,
            rotations_gt=rotations_gt,
            translations_gt=translations_gt,
            camera_positions_world=camera_positions_world,
            R₁=R₁,
            start_sol=start_sol,
            target_sol=target_sol,
            n_lines=n_lines,
            lines_view1=lines_views[1],
            lines_view2=lines_views[2],
            calibrated_matrices=calibrated_matrices,
        )
    end

    """
        track_with_steps(tracker, start_sol; max_iters=20000)
        track_with_steps(homotopy, start_sol; max_iters=20000)

    Run HC's low-level `Tracker` manually with `step!` so we can record `(t, x)`
    after every accepted step. Returns `(steps::Vector{Tuple{Float64,Vector{ComplexF64}}}, code::TrackerCode.codes)`.
    The homotopy variant builds a fresh `Tracker`; the tracker variant lets the
    caller reuse an existing one (useful for the total-degree branch, where we
    only want one `total_degree(...)` call).
    """
    function track_with_steps(tracker::Tracker, start_sol::AbstractVector{<:Number}; max_iters::Int=20000)
        HomotopyContinuation.init!(tracker, Vector{ComplexF64}(start_sol), 1.0, 0.0)
        steps = Vector{Tuple{Float64, Vector{ComplexF64}}}()
        push!(steps, (real(tracker.state.t), copy(tracker.state.x)))
        iter = 0
        while HomotopyContinuation.is_tracking(tracker.state.code) && iter < max_iters
            HomotopyContinuation.step!(tracker)
            push!(steps, (real(tracker.state.t), copy(tracker.state.x)))
            iter += 1
        end
        return steps, tracker.state.code
    end

    function track_with_steps(homotopy::AbstractHomotopy, start_sol::AbstractVector{<:Number}; max_iters::Int=20000)
        return track_with_steps(Tracker(homotopy), start_sol; max_iters=max_iters)
    end

    """
        _search_total_degree_best(inner_tracker, starts_iter, target_sol_c; limit, close_enough)

    Scan up to `limit` of the Bezout start solutions, tracking each with the
    straight-line tracker inside `endgame_total.tracker`. Returns the start
    solution whose final x is closest to `target_sol_c`, or `nothing` if none
    converged within `limit` tries.
    """
    function _search_total_degree_best(inner_tracker::Tracker, starts_iter,
                                       target_sol_c::Vector{ComplexF64};
                                       limit::Int=4096, close_enough::Float64=1e-3)
        best_dist = Inf
        best_start = nothing
        tried = 0
        for start in starts_iter
            tried += 1
            tried > limit && break
            start_v = Vector{ComplexF64}(start)
            try
                HomotopyContinuation.track!(inner_tracker, start_v, 1.0, 0.0)
                if HomotopyContinuation.is_success(inner_tracker.state.code)
                    dist = norm(inner_tracker.state.x - target_sol_c)
                    if dist < best_dist
                        best_dist = dist
                        best_start = start_v
                        if dist < close_enough
                            return best_start, best_dist, tried
                        end
                    end
                end
            catch
                continue
            end
        end
        return best_start, best_dist, tried
    end

    """
        run_four_homotopies_tracking(; random_seed=84564, save_path, total_degree_search_limit, total_degree_gamma_seed)

    Build the toy problem, construct the four homotopies, step through each
    with the HC `Tracker`, and save every per-step `(t, x)` snapshot to
    `save_path` (a `.jls` file).

    `total_degree_search_limit` caps how many Bezout start solutions we try
    when looking for one that reaches the known target camera. The full
    Bezout number is 4⁶ = 4096 — for a demo the first few hundred usually
    hit a converging path.
    """
    function run_four_homotopies_tracking(;
            random_seed::Int=84564,
            save_path::String=default_tracked_path,
            total_degree_search_limit::Int=4096,
            total_degree_gamma_seed::Int=42)

        setup = setup_toy_problem(random_seed=random_seed)

        @info "Residual at start solution: $(norm(setup.F(setup.start_sol, setup.p)))"
        @info "Residual at target solution: $(norm(setup.F(setup.target_sol, setup.q)))"

        tracked_paths = Dict{String, NamedTuple}()

        @info "Tracking InfiniteHomographyHomotopy..."
        inf_hom = InfiniteHomographyHomotopy(setup.F, setup.p, setup.q, setup.Hinf, setup.vps_view1)
        inf_steps, inf_code = track_with_steps(inf_hom, setup.start_sol)
        @info "  → $(length(inf_steps)) steps, code=$inf_code"
        tracked_paths["InfiniteHomography"] = (steps=inf_steps, code=inf_code)

        @info "Tracking GeometricHomotopy..."
        geo_hom = GeometricHomotopy(setup.F, setup.p, setup.q)
        geo_steps, geo_code = track_with_steps(geo_hom, setup.start_sol)
        @info "  → $(length(geo_steps)) steps, code=$geo_code"
        tracked_paths["Geometric"] = (steps=geo_steps, code=geo_code)

        @info "Tracking HC's ParameterHomotopy (straight-line in parameter space)..."
        par_hom = HomotopyContinuation.ParameterHomotopy(setup.F; start_parameters=Vector{ComplexF64}(setup.p), target_parameters=Vector{ComplexF64}(setup.q))
        par_steps, par_code = track_with_steps(par_hom, setup.start_sol)
        @info "  → $(length(par_steps)) steps, code=$par_code"
        tracked_paths["Parameter"] = (steps=par_steps, code=par_code)

        @info "Running total degree homotopy..."
        Random.seed!(total_degree_gamma_seed)
        γ_total = cis(2π * rand())
        endgame_total, starts_iter = total_degree(setup.F; target_parameters=setup.q, γ=γ_total)
        inner_total = endgame_total.tracker
        target_sol_c = Vector{ComplexF64}(setup.target_sol)

        @info "  Searching best Bezout start (limit=$total_degree_search_limit)..."
        best_start, best_dist, tried = _search_total_degree_best(
            inner_total, starts_iter, target_sol_c;
            limit=total_degree_search_limit, close_enough=1e-3)
        @info "  Scanned $tried paths; best distance to target = $best_dist"

        if best_start === nothing
            @warn "Total degree did not find any converging path within limit"
            tracked_paths["TotalDegree"] = (steps=nothing, code=:not_found)
        else
            @info "Re-tracking best total degree path to collect steps..."
            total_steps, total_code = track_with_steps(inner_total, best_start)
            @info "  → $(length(total_steps)) steps, code=$total_code, best_dist=$best_dist"
            tracked_paths["TotalDegree"] = (steps=total_steps, code=total_code)
        end

        serializable_setup = (
            vanishing_points_3d=setup.vanishing_points_3d,
            points_3d=setup.points_3d,
            intrinsics=setup.intrinsics,
            R₁=setup.R₁,
            rotations_gt=setup.rotations_gt,
            translations_gt=setup.translations_gt,
            camera_positions_world=setup.camera_positions_world,
            start_sol=setup.start_sol,
            target_sol=setup.target_sol,
            lines_view1=setup.lines_view1,
            lines_view2=setup.lines_view2,
            n_lines=setup.n_lines,
            calibrated_matrices=setup.calibrated_matrices,
        )

        data = (setup=serializable_setup, paths=tracked_paths)
        mkpath(dirname(save_path))
        serialize(save_path, data)
        @info "Saved tracked paths to $save_path"
        return data
    end

    """
        _closest_step_index(steps, t_target)

    Linear scan — the step vectors are at most a few hundred entries, so this is
    plenty fast and avoids an extra dependency on a sorted search helper.
    """
    function _closest_step_index(steps::Vector{Tuple{Float64, Vector{ComplexF64}}}, t_target::Real)
        best_idx = 1
        best_diff = Inf
        @inbounds for i in eachindex(steps)
            d = abs(steps[i][1] - t_target)
            if d < best_diff
                best_diff = d
                best_idx = i
            end
        end
        return best_idx
    end

    # ───────────────────── GLMakie-dependent visualization ─────────────────────
    #
    # The function below only works if GLMakie is loaded. We define it in a
    # separate code block so that `include`-ing this file never requires a
    # display, mirroring the Plotting module's conditional loading pattern.

    if GUI_ENABLED && Base.find_package("GLMakie") !== nothing
        include("visualizations-makie.jl")
    else
        function visualize_four_homotopies(args...; kwargs...)
            error("visualize_four_homotopies requires GLMakie (GUI_ENABLED=true)")
        end
    end
end
