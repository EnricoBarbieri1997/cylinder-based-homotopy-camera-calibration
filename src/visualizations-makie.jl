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
    _slider_to_step(s, pivot, N)
    _step_to_slider(step, pivot, N)

Piecewise-linear remapping between slider position `s ∈ [0, N]` and step index
`step ∈ [0, N]` with a user-chosen `pivot ∈ (0, N)` that lands at the slider
midpoint `N/2`. When `pivot == N/2` the map is the identity. Lower `pivot`
zooms the slider's left half into the low step indices; higher `pivot` zooms
the right half into the high step indices.
"""
function _slider_to_step(s::Real, pivot::Real, N::Real)
    pivot = clamp(float(pivot), 1.0, float(N) - 1.0)
    half = N / 2
    return s <= half ?
           s * pivot / half :
           pivot + (s - half) * (N - pivot) / half
end
function _step_to_slider(step::Real, pivot::Real, N::Real)
    pivot = clamp(float(pivot), 1.0, float(N) - 1.0)
    half = N / 2
    return step <= pivot ?
           step * half / pivot :
           half + (step - pivot) * half / (N - pivot)
end

"""
    visualize_four_homotopies(data_path=default_tracked_path)

Interactive comparison of the four homotopy paths saved by
`run_four_homotopies_tracking`. The slider at the bottom indexes directly
into HC's recorded step list (0 → initial state, max → final step of the
longest-running homotopy). Homotopies that stopped early freeze at their
last recorded step, and a colored dot above the slider marks where each
one actually stopped. Start and target ground-truth poses are always drawn
in light gray and black respectively.
"""
function visualize_four_homotopies(data_path::String=default_tracked_path)
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

    # Max step count across all homotopies defines the slider range. Each
    # homotopy clamps to its own last step when the slider is past its length.
    max_steps = 0
    for name in HOMOTOPY_ORDER
        info = paths[name]
        info.steps === nothing && continue
        max_steps = max(max_steps, length(info.steps))
    end
    max_steps == 0 && error("All homotopy paths are empty — nothing to visualize")

    fig = Figure(size=(1600, 900))

    ax3d = Axis3(fig[1, 1]; aspect=:data, title="Camera trajectories (world coords)")

    grid_right = fig[1, 2] = GridLayout()
    axes_2d = Dict{String, Any}()
    for (i, name) in enumerate(HOMOTOPY_ORDER)
        color = HOMOTOPY_COLORS[name]
        row = ((i - 1) ÷ 2) + 1
        col = ((i - 1) % 2) + 1
        ax = Axis(grid_right[row, col]; title=name, titlecolor=color,
                  aspect=DataAspect(), yreversed=true)
        axes_2d[name] = ax
    end
    colsize!(fig.layout, 1, Relative(0.62))
    colsize!(fig.layout, 2, Relative(0.38))

    # Bottom rows hold the stop-dot indicator, the main step slider, and the
    # pivot slider that controls how step indices map onto slider positions.
    slider_box = fig[2, 1:2] = GridLayout()
    # Row 1: dot strip, aligned over the slider column (col 2 of slider_box).
    dot_strip = Axis(slider_box[1, 2]; height=22)
    hidespines!(dot_strip)
    hidedecorations!(dot_strip)
    xlims!(dot_strip, 0, max_steps)
    ylims!(dot_strip, -1, 1)
    # Row 2: main slider (slider position → remapped step index).
    Label(slider_box[2, 1], "Step:"; halign=:right)
    slider = Slider(slider_box[2, 2]; range=0:1:max_steps, startvalue=0)
    # Row 3: pivot slider (controls the remapping; default at the midpoint
    # gives a 1-to-1 identity mapping).
    Label(slider_box[3, 1], "Pivot:"; halign=:right)
    pivot_slider = Slider(slider_box[3, 2]; range=1:1:(max_steps-1), startvalue=max_steps ÷ 2)

    # Current remapped step index (float, used both for display and reactive x).
    current_step = lift(slider.value, pivot_slider.value) do s, pivot
        _slider_to_step(s, pivot, max_steps)
    end
    Label(slider_box[2, 3],
          lift(current_step) do step
              "step $(round(Int, step)) / $max_steps"
          end;
          halign=:left)
    Label(slider_box[3, 3],
          lift(pivot_slider.value) do p
              "$p / $max_steps"
          end;
          halign=:left)

    # Stop-dots: positions depend on the pivot slider.
    for name in HOMOTOPY_ORDER
        info = paths[name]
        info.steps === nothing && continue
        color = HOMOTOPY_COLORS[name]
        nsteps_this = length(info.steps)
        dot_pos = lift(pivot_slider.value) do pivot
            x = _step_to_slider(nsteps_this, pivot, max_steps)
            return [Point2f(x, 0.0)]
        end
        scatter!(dot_strip, dot_pos; color=color, markersize=14)
    end

    # Arrow-key navigation on the main slider: ← and → step by 1,
    # Shift+← / Shift+→ step by 10 for faster scrubbing.
    on(events(fig).keyboardbutton) do event
        if event.action in (Keyboard.press, Keyboard.repeat)
            shift = ispressed(fig.scene, Keyboard.left_shift) ||
                    ispressed(fig.scene, Keyboard.right_shift)
            delta = shift ? 10 : 1
            cur = slider.value[]
            new_val = cur
            if event.key == Keyboard.left
                new_val = max(0, cur - delta)
            elseif event.key == Keyboard.right
                new_val = min(max_steps, cur + delta)
            end
            if new_val != cur
                set_close_to!(slider, new_val)
            end
        end
        return Consume(false)
    end

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
        nsteps = length(steps)

        # The main slider is remapped through the pivot slider into a step
        # index; we round and clamp into THIS homotopy's recorded steps so
        # paths that stopped early freeze at their final pose.
        current_x = lift(current_step) do step_f
            idx = clamp(round(Int, step_f) + 1, 1, nsteps)
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
