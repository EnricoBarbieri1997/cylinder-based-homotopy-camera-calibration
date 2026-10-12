# GLMakie-dependent visualization for `visualize_four_homotopies`.
# Loaded by `visualizations.jl` only when GUI_ENABLED and GLMakie is installed.

using GLMakie
using GeometryBasics: Point2f, Point3f

"""
    _project_line_segment_2d(R, t, K, w, v; half_length=10.0)

Project a 3-D line (passing through `w` with direction `v`) through the camera
`(R, t)` with intrinsic `K`, returning two 2-D points on it in pixel space.
Returns `nothing` if either endpoint is behind the camera (so the caller can
draw a NaN segment that Makie skips).
"""
function _project_line_segment_2d(R::AbstractMatrix{<:Real}, t::AbstractVector{<:Real},
                                  K::AbstractMatrix{<:Real},
                                  w::AbstractVector{<:Real}, v::AbstractVector{<:Real};
                                  half_length::Real=10.0)
    p1_3d = w .+ half_length .* v
    p2_3d = w .- half_length .* v
    p1_cam = R * p1_3d .+ t
    p2_cam = R * p2_3d .+ t
    if p1_cam[3] <= 1e-6 || p2_cam[3] <= 1e-6
        return nothing
    end
    p1_img = K * p1_cam
    p2_img = K * p2_cam
    return (Point2f(p1_img[1] / p1_img[3], p1_img[2] / p1_img[3]),
            Point2f(p2_img[1] / p2_img[3], p2_img[2] / p2_img[3]))
end

"""
    _draw_camera_marker!(ax, pos, R, color; forward_length=2.0, markersize=18)

Draw a scatter point for the camera center plus a short line segment showing
its forward direction (R' · [0,0,1] in world coords).
"""
function _draw_camera_marker!(ax, pos::AbstractVector{<:Real}, R::AbstractMatrix{<:Real}, color;
                              forward_length::Real=2.0, markersize::Real=18, label::String="")
    scatter!(ax, Point3f(pos[1], pos[2], pos[3]); color=color, markersize=markersize)
    forward = R' * [0.0, 0.0, 1.0]
    tip = pos .+ forward_length .* forward
    lines!(ax, [Point3f(pos[1], pos[2], pos[3]), Point3f(tip[1], tip[2], tip[3])];
           color=color, linewidth=3)
    if !isempty(label)
        text!(ax, label; position=Point3f(pos[1], pos[2], pos[3] + 0.5), color=color, align=(:center, :bottom))
    end
end

"""
    _static_lines_in_panel!(ax, R, t, K, points_3d, vps_3d, color)

Project each 3-D scene line through camera `(R, t, K)` and draw it in `ax`.
Used for the "start" and "target" ground-truth lines (light gray / black).
"""
function _static_lines_in_panel!(ax, R, t, K, points_3d, vps_3d, color; linewidth=1.5, linestyle=:solid)
    for i in eachindex(points_3d)
        seg = _project_line_segment_2d(R, t, K, points_3d[i], vps_3d[i])
        seg === nothing && continue
        lines!(ax, [seg[1], seg[2]]; color=color, linewidth=linewidth, linestyle=linestyle)
    end
end

"""
    visualize_four_homotopies(data_path=default_tracked_path; progress_steps=100)

Interactive comparison of the four homotopy paths saved by
`run_four_homotopies_tracking`. The slider at the bottom moves each camera
from its start pose (`t=1`, progress 0) toward its target pose (`t=0`,
progress 100). The start and target poses are always drawn in light gray and
black respectively.
"""
function visualize_four_homotopies(data_path::String=default_tracked_path; progress_steps::Int=100)
    isfile(data_path) || error("No tracked-paths file at $data_path — run `run_four_homotopies_tracking()` first")
    data = deserialize(data_path)
    setup = data.setup
    paths = data.paths

    R_start = setup.rotations_gt[1]
    R_target = setup.rotations_gt[2]
    t_start = setup.translations_gt[1]
    t_target = setup.translations_gt[2]
    pos_start = -R_start' * t_start
    pos_target = -R_target' * t_target
    K = setup.intrinsics
    K_norm = K ./ K[2, 2]

    fig = Figure(size=(1600, 900))

    ax3d = Axis3(fig[1, 1]; aspect=:data, title="Camera trajectories (world coords)")

    grid_right = fig[1, 2] = GridLayout()
    axes_2d = Dict{String, Any}()
    for (i, name) in enumerate(HOMOTOPY_ORDER)
        color = HOMOTOPY_COLORS[name]
        ax = Axis(grid_right[i, 1]; title=name, titlecolor=color,
                  aspect=DataAspect(), yreversed=true)
        axes_2d[name] = ax
    end
    colsize!(fig.layout, 1, Relative(0.62))
    colsize!(fig.layout, 2, Relative(0.38))

    slider_box = fig[2, 1:2] = GridLayout()
    Label(slider_box[1, 1], "Progress (0 = start, $progress_steps = target):"; halign=:right)
    slider = Slider(slider_box[1, 2]; range=0:1:progress_steps, startvalue=0)
    Label(slider_box[1, 3], lift(v -> "$v / $progress_steps", slider.value); halign=:left)

    # ───── Static 3-D content: 3-D points, 3-D lines, start/target cameras ─────
    for w in setup.points_3d
        scatter!(ax3d, Point3f(w[1], w[2], w[3]); color=:dimgray, markersize=10)
    end
    for i in 1:setup.n_lines
        w = setup.points_3d[i]
        v = setup.vanishing_points_3d[i]
        half = 4.0
        p1 = w .+ half .* v
        p2 = w .- half .* v
        lines!(ax3d, [Point3f(p1[1], p1[2], p1[3]), Point3f(p2[1], p2[2], p2[3])];
               color=:lightslategray, linewidth=1)
    end
    _draw_camera_marker!(ax3d, pos_start, R_start, :lightgray; label="start")
    _draw_camera_marker!(ax3d, pos_target, R_target, :black; label="target")

    # ───── Static 2-D content per panel: start lines (light gray), target (black) ─────
    for name in HOMOTOPY_ORDER
        ax = axes_2d[name]
        _static_lines_in_panel!(ax, R_start, t_start, K_norm,
                                setup.points_3d, setup.vanishing_points_3d,
                                :lightgray; linestyle=:dash)
        _static_lines_in_panel!(ax, R_target, t_target, K_norm,
                                setup.points_3d, setup.vanishing_points_3d,
                                :black; linestyle=:solid)
    end

    # ───── Reactive per-homotopy drawing ─────
    for name in HOMOTOPY_ORDER
        info = paths[name]
        info.steps === nothing && continue
        color = HOMOTOPY_COLORS[name]
        steps = info.steps

        # Reactive solution at current slider position
        current_x = lift(slider.value) do progress
            t = 1.0 - progress / progress_steps
            idx = _closest_step_index(steps, t)
            return real.(steps[idx][2])
        end

        # Reactive pose (R, t_vec, pos_world) from solution
        current_pose = lift(current_x) do xs
            R, tv = solution_to_pose(xs, setup.R₁)
            pos = -R' * tv
            return (R=R, t=tv, pos=pos)
        end

        current_point = lift(cp -> Point3f(cp.pos[1], cp.pos[2], cp.pos[3]), current_pose)
        current_forward_tip = lift(current_pose) do cp
            tip = cp.pos .+ 2.0 .* (cp.R' * [0.0, 0.0, 1.0])
            return Point3f(tip[1], tip[2], tip[3])
        end

        scatter!(ax3d, current_point; color=color, markersize=18)
        lines!(ax3d, lift((p, f) -> [p, f], current_point, current_forward_tip);
               color=color, linewidth=3)

        # Camera trail: polyline of all step positions, dimmed
        step_positions = Point3f[]
        for (_, xs) in steps
            R, tv = solution_to_pose(real.(xs), setup.R₁)
            pos = -R' * tv
            push!(step_positions, Point3f(pos[1], pos[2], pos[3]))
        end
        lines!(ax3d, step_positions; color=color, linewidth=1, alpha=0.4)

        ax2 = axes_2d[name]
        for i in 1:setup.n_lines
            seg_obs = lift(current_pose) do cp
                seg = _project_line_segment_2d(cp.R, cp.t, K_norm,
                                               setup.points_3d[i],
                                               setup.vanishing_points_3d[i])
                if seg === nothing
                    return [Point2f(NaN, NaN), Point2f(NaN, NaN)]
                end
                return [seg[1], seg[2]]
            end
            lines!(ax2, seg_obs; color=color, linewidth=2)
        end
    end

    display(fig)
    return fig
end
