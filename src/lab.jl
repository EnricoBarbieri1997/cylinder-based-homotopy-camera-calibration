module Lab
    using HomotopyContinuation
    using DelimitedFiles
    using StatsBase
    using LinearAlgebra: norm, normalize, dot, cross
    using Rotations

    using ..Scene: SceneData, InstanceConfiguration, intrinsic_rotation_system_setup
    using ..Camera: CameraProperties, CameraViewPair, random_camera_lookingat_center, lookat_quaternion
    using ..Geometry: get_view, Line, compute_Hinf
    using ..IO: read_point_cloud_zup, read_cylinders_zup, read_camera_from_matrices
    using ..Plotting: initfigure, plot_3dcylinders, plot_3dpoints, plot_3dcamera
    using ..Cylinder: cylinder_from_center_axis_radius, points_at_infinity_dualquadrics
    using ..Cylinder.CalibrationRigs: arbitrary_rig, arbitrary_rig_four
    using ..EquationSystems.Problems: CylinderCameraContoursProblem, CylinderCameraContoursProblemValidationData
    using ..EquationSystems.Problems.IntrinsicParameters: Configurations as IntrinsicParametersConfigurations
    using ..Utils: rand_in_range, lines_clp_to_stack
    using ..Homotopies: GeometricHomotopy, InfiniteHomographyHomotopy, interpolate_line, verify_line_through_vanishing_point
    using ..Homotopies
    using ..Printing: print_scene_config_results, scene_config_results_table_data

    using Random
    using Statistics
    using Serialization

    function shuttercameras()
        link1 = "https://gist.githubusercontent.com/PBrdng/e17d0e3bc4d983734238b9cb8386d560/raw/07272b125a6ad03c791fdf99e741318f1d85149b/3Dpoints"
        link2 = "https://gist.githubusercontent.com/PBrdng/e17d0e3bc4d983734238b9cb8386d560/raw/07272b125a6ad03c791fdf99e741318f1d85149b/2Dpoints"
        points3D = readdlm(download(link1)) |> transpose
        points2D = readdlm(download(link2)) |> transpose
        τ = 0.01
        m = 5
        @var x[1:3, 1:m], u[1:2, 1:m]
        @var R[1:3, 1:3], t[1:3], v[1:3], n[1:m]

        cams = [[R t-(n[i]*τ).*v] for i in 1:m] # m = 5 cameras

        y = [cams[i] * [x[:,i]; 1] for i in 1:m] # m = 5 images

        g = [cross(y[i],  [u[:,i]; 1]) for i in 1:m]

        k = R * R' - diagm(ones(3))
        Rconstraints = [k[i,j] for i in 1:3, j in 1:3 if i<=j]
        vconstraints = transpose(v) * v - 1

        @var l1[1:6], l2       # Lagrange multipliers 

        L = transpose(g) * g - transpose(l1) * Rconstraints - l2 * vconstraints
        Lag = differentiate(L, [vec(R); t; v; l1; l2]);

        F = System(Lag, variables = [vec(R); t; v; l1; l2], parameters = [vec(x); vec([u; n'])]);
        display(F)

        p0 = randn(ComplexF64, 30) 
        S0 = solve(F, target_parameters = p0)
        start = solutions(S0);

        N = size(points2D, 2)
        s = StatsBase.sample(collect(1:N), m; replace=false)

        X = points2D[1:3, s] 
        Y = points2D[:, s]

        p1 = [vec(X); vec(Y)]
        S1 = solve(F, start, start_parameters = p0, target_parameters = p1)
        G = System(vcat(g...), variables = [vec(R); t; v], parameters = [vec(x); vec([u; n'])]);
        function find_min(sols, p)
            sols = [r[1:15] for r in sols]
            a = map(sols) do r 
                norm(G(r, p))
            end
            i = findmin(a)
            sols[i[2]]
        end
        recovery = find_min(real_solutions(S1), p1)
        R1, t1, v1 = reshape(recovery[1:9], 3, 3), recovery[10:12], recovery[13:15]
    end

    function dinosaur()
        include(download("https://gist.githubusercontent.com/PBrdng/46436855f3755c5a959a7c5d6ba7e32b/raw/5e6be8f4c9673f0dd26010b5e0cefc8e953fc1c0/cameras.jl"))
        include(download("https://gist.githubusercontent.com/PBrdng/46436855f3755c5a959a7c5d6ba7e32b/raw/5e6be8f4c9673f0dd26010b5e0cefc8e953fc1c0/pictures.jl"))
        n₁, n₂ = 1, 2
        camera_numbers = [n₁, n₂]

        @var x[1:3]
        @var t[1:2]
        @var p[1:2,1:2]

        y = [cams[i][1:2,:]*[x; 1] for i in camera_numbers]
        z = [cams[i][3,:] ⋅ [x; 1] for i in camera_numbers] .* 648

        g = [t[i] .* y[i] - p[:,i] for i in 1:2]
        G = sum(gᵢ ⋅ gᵢ for gᵢ in g)
        # G is the function that we want to optimize

        @var λ[1:2] # the Lagrance multipliers
        L = G - sum(λ[i] * (t[i] * z[i] - 1) for i in 1:2)
        ∇L = differentiate(L, [x;t;λ])
        F = System(∇L; variables = [x;t;λ], parameters = vec(p))
        display(F)

        p₀ = randn(ComplexF64, 4)
        start = solutions(solve(F; target_parameters = p₀))

        #first, we need to preprocess the photo data
        photos = [ps[i, [2 * n₁ - 1, 2 * n₁, 2 * n₂ - 1, 2 * n₂]] for i in 1:4983]

        # the data from the dataset is incomplete.
        # the cameras did not take pictures of all world points.
        # if a world point was not captured, the entry in the data set is -1.
        filter!(pᵢ -> all(pᵢ .> 0), photos)

        # we divide the photo coordinates by 648
        # to work with coordinates between 0 and 1
        # (as explained above)
        map!(pᵢ -> pᵢ./648, photos, photos)

        function reconstruction_from_critical_points(R, pᵢ, G)
            r = real_solutions(R)
            N = map(r) do rᵢ
                G([x;t] => rᵢ[1:5], vec(p) => pᵢ)
            end
            i = findmin(N)

            return r[i[2]][1:3]
        end

        reconstructed_points = solve(
            F,
            start;
            start_parameters =  p₀,
            target_parameters = photos,
            transform_result = (R,pᵢ) -> reconstruction_from_critical_points(R,pᵢ,G)
        )
    end

    function ellipses_meet()
        @polyvar Q₁[1:2, 1:2] Q₂[1:2, 1:2] p₁[1:2] p₂[1:2]
        @polyvar x[1:2] r
        z₁ = x - p₁
        z₂ = x - p₂
        # initialize the equations for E₁ and E₂
        f₁ = (Q₁ * z₁) ⋅ (Q₁ * z₁) - r^2
        f₂ = (Q₂ * z₂) ⋅ (Q₂ * z₂) - r^2
        # initialize the equation for E₁ and E₂ being tangent
        @polyvar λ
        g = (Q₁' * Q₁) * z₁ - λ .* (Q₂' * Q₂) * z₂
        # gather everything in one system
        params = [vec(Q₁); vec(Q₂); p₁; p₂]
        F = System([f₁; f₂; g]; variables=[x; r; λ], parameters=params)
        display(F)

        q = [1, 0, 0, 1, 1, 0, 0, 1, 1, 0, -1, 0]
        p = [vec([1 1; 1 0]); vec([0 2; 2 1]); [3, 0]; [1, 2]]
        solve(F, [[0, 0, 1, -1]], start_parameters=q, target_parameters=p)
    end

    function tritangents()
        @var h[1:3] # variables for the plane
        @var x[1:3] y[1:3] z[1:3] #variables for the contact points

        #the quadric
        Q = x[3] - x[1] * x[2]
        #the cubic C with coefficients c
        C, c = dense_poly(x, 3, coeff_name = :c)

        #generate the system P for the contact point x
        P_x = [
        h ⋅ x - 1;
        Q;
        C;
        det([h differentiate(Q, x) differentiate(C, x)])
        ]

        #generate a copy of P for the other contact points y,z
        P_y = [p([h; x; c] => [h; y; c]) for p in P_x]
        P_z = [p([h; x; c] => [h; z; c]) for p in P_x]

        #define F
        F = System([P_x; P_y; P_z]; variables = [h;x;y;z], parameters = c)
        display(F)

        c₁ = randn(ComplexF64, 20)
        #solve the system for c₁
        S = solve(F; target_parameters = c₁)

        sols = solutions(S)

        #define the coefficients for C
        c₀ = coeffs_as_dense_poly(x[1]^3+x[2]^3+x[3]^3-1, x, 3)
        #track the solutions from c₁ to c₀
        R = solve(F, sols, start_parameters = c₁, target_parameters = c₀)
        display(R)
    end

    function circles_tangent_to_conics()
        @var a[1:2] r #variables for the circle center and radius
        @var x y #variables of the circle
        @var B[1:3,1:3] #coefficients of the conics
        @var v[1:2, 1:3] #variables of the 3 points at which the circle is tangent

        circle = ([x; y] - a) ⋅ ([x; y] - a) - r
        conic  = [x; y; 1] ⋅ (B * [x; y; 1]);
        tangential_condition = det([differentiate(circle, [x, y]) differentiate(conic, [x, y])])

        conditions = [circle; conic; tangential_condition]

        #define coefficients of the three conics
        C1 = randn(3,3)
        C2 = randn(3,3)
        C3 = randn(3,3)

        #Plug in the variables of the 3 points
        #and coefficients of the 3 conics
        F = System([
            f([x; y; a; r; vec(B)] => [v[:,i]; a; r; vec(C)])
            for f in conditions
            for (i,C) in enumerate([C1, C2, C3])
            ])
        display(F)
        sol = solve(F)
        display(sol)
    end

    function view_tmp_point_cloud_cylinders()
        filepath = "./assets/test_scenes/water_tower/scene.json"
        points = read_point_cloud_zup(filepath)
        cylinders = read_cylinders_zup(filepath)
        camera1 = read_camera_from_matrices(
            filepath,
            1
        )
        camera2 = read_camera_from_matrices(
            filepath,
            79
        )

        figure = initfigure()
        plot_3dpoints(points; color=:black, markersize=4)
        plot_3dcylinders([
            cylinder_from_center_axis_radius(cylinder.center, cylinder.axis, cylinder.radius)
            for cylinder in cylinders
        ])
        plot_3dcamera(camera1, :red)
        plot_3dcamera(camera2, :blue)
        display(figure)

        # return points, cylinders
    end

    function problems_from_scene(scene; intrinsic_configuration=IntrinsicParametersConfigurations.fₓ_fᵧ_cₓ_cᵧ)
        points_at_infinity, dualquadrics = points_at_infinity_dualquadrics(scene.cylinders)
                
        problems::Vector{CylinderCameraContoursProblem} = []
        validation_data = CylinderCameraContoursProblemValidationData(
            Matrix{Float64}(undef, 0, 3),
            Matrix{Float64}(undef, 0, 3),
            Array{Float64}(undef, 0, 4, 4),
        )
        for (_, instance) in enumerate(scene.instances)
            lines = lines_clp_to_stack(instance.conics_contours)
            problem = CylinderCameraContoursProblem(
                CameraProperties(),
                lines,
                lines,
                points_at_infinity,
                dualquadrics,
                validation_data,
                UInt8(intrinsic_configuration)
            )
            push!(problems, problem)
        end

        return problems
    end

    function compare_execution_times()
        Random.seed!(777)
        intrinsic_configuration = IntrinsicParametersConfigurations.fₓ_fᵧ_cₓ_cᵧ

        scenes = []
        true_solutions = []

        cylinder_view_configurations = reverse([
            Dict(
                "number_of_cylinders" => 2,
                "number_of_views" => 4,
                "delta_fx" => [],
                "delta_fy" => [],
                "delta_cx" => [],
                "delta_cy" => [],
                "delta_rotation" => [],
                "elapsed_time" => [],
            ),
            Dict(
                "number_of_cylinders" => 3,
                "number_of_views" => 2,
                "delta_fx" => [],
                "delta_fy" => [],
                "delta_cx" => [],
                "delta_cy" => [],
                "delta_rotation" => [],
                "elapsed_time" => [],
            ),
            Dict(
                "number_of_cylinders" => 4,
                "number_of_views" => 1,
                "delta_fx" => [],
                "delta_fy" => [],
                "delta_cx" => [],
                "delta_cy" => [],
                "delta_rotation" => [],
                "elapsed_time" => [],
            ),
        ])

        for _ in 1:8
            intrinsics = [
                rand_in_range(2500.0, 2700.0) 0.0 rand_in_range(950.0, 970.0);   # fₓ, skew, cₓ
                0.0 rand_in_range(1400.0, 1600.0) rand_in_range(530.0, 550.0);   # 0, fᵧ, cᵧ
                0.0 0.0 1.0                                                      # bottom row
            ]

            cylinders = arbitrary_rig_four()

            cameras = []
            for _ in 1:4
                position, rotation = random_camera_lookingat_center()
                camera = CameraProperties()
                camera.position = position
                camera.quaternion_rotation = rotation
                camera.intrinsic = intrinsics
                push!(cameras, camera)
            end

            scene = SceneData()
            scene.figure = initfigure()
            scene.cylinders = cylinders
            scene.instances = map(camera -> begin
                instance = InstanceConfiguration()
                instance.camera = camera
                return instance
            end, cameras)

            push!(scenes, scene)
        end

        #region true solutions for original problems
        for scene in scenes
            true_intrinsics = scene.instances[1].camera.intrinsic ./ 3000.0
            rotations = map(instance -> begin
                rot = Rotations.params(QuatRotation(instance.camera.rotation_matrix))
                rot = rot / rot[1]
                rot = rot[2:4]
                return rot
            end, scene.instances)

            # Merge all rotations into a single vector
            merged_rotations = vcat(rotations...)

            true_solution = [
                true_intrinsics[1, 1],
                true_intrinsics[2, 2],
                true_intrinsics[1, 3],
                true_intrinsics[2, 3],
                merged_rotations...,
            ]
            push!(true_solutions, true_solution)
        end
        #endregion

        offset = 2.0

        for config in cylinder_view_configurations
            delta_fx = []
            delta_fy = []
            delta_cx = []
            delta_cy = []
            delta_rotation = []
            elapsed_time = []

            solved_instances_line_parametric = 0
            solved_paths_line_parametric = 0
            tracked_paths_line_parametric = 0

            number_of_cylinders = config["number_of_cylinders"]
            number_of_views = config["number_of_views"]

            for (i, _scene) in enumerate(scenes)
                display("Processing scene: $i, configuration: $(config["number_of_cylinders"]) cylinders, $(config["number_of_views"]) views")

                scene = SceneData()
                scene.figure = initfigure()
                scene.cylinders = _scene.cylinders[1:number_of_cylinders]
                scene.instances = deepcopy(_scene.instances[1:number_of_views])
                for instance in scene.instances
                    instance.conics_contours = get_view(scene.cylinders, instance.camera)
                end
                problems = problems_from_scene(scene)
                _, parameters = intrinsic_rotation_system_setup(problems; intrinsic_configuration)

                offsetted_scene = SceneData()
                offsetted_scene.figure = initfigure()
                offsetted_scene.cylinders = scene.cylinders

                offsetted_intrinsics = scene.instances[1].camera.intrinsic + [
                    (rand(Float64) * 50.0 - 25.0) 0.0 (rand(Float64) * 10.0 - 5.0);   # fₓ, skew, cₓ
                    0.0 (rand(Float64) * 50.0 - 25.0) (rand(Float64) * 10.0 - 5.0);   # 0, fᵧ, cᵧ
                    0.0 0.0 0.0                                                      # bottom row
                ]

                offsetted_scene.instances = map((instance) -> begin
                    offsetted_camera = CameraProperties()
                    offsetted_camera.intrinsic = offsetted_intrinsics
                    random_direction = normalize(randn(3))
                    offsetted_camera.position = instance.camera.position + offset * random_direction
                    rotation = lookat_quaternion(offsetted_camera.position, [0.0, 0.0, 0.0])
                    offsetted_camera.quaternion_rotation = rotation
                    view = get_view(offsetted_scene.cylinders, offsetted_camera)
                    instance = InstanceConfiguration()
                    instance.camera = offsetted_camera
                    instance.conics_contours = view
                    return instance
                end, scene.instances)

                offsetted_problems = problems_from_scene(offsetted_scene)

                rotation_intrinsic_system, offsetted_parameters = intrinsic_rotation_system_setup(offsetted_problems;
                    intrinsic_configuration
                )

                #region Computing original sol
                rotations = map(instance -> begin
                    rot = Rotations.params(QuatRotation(instance.camera.rotation_matrix))
                    rot = rot / rot[1]
                    rot = rot[2:4]
                    return rot
                end, offsetted_scene.instances)

                # Merge all rotations into a single vector
                merged_rotations = vcat(rotations...)
                known_solution = [
                    offsetted_scene.instances[1].camera.intrinsic[1, 1] / 3000.0,
                    offsetted_scene.instances[1].camera.intrinsic[2, 2] / 3000.0,
                    offsetted_scene.instances[1].camera.intrinsic[1, 3] / 3000.0,
                    offsetted_scene.instances[1].camera.intrinsic[2, 3] / 3000.0,
                    merged_rotations...,
                ]
                #endregion

                monodromy_solutions = solutions(
                    monodromy_solve(
                        rotation_intrinsic_system,
                        [known_solution],
                        offsetted_parameters;
                        show_progress=false,
                    )
                )

                line_parametric_result = nothing
                t = @elapsed begin
                    line_parametric_result = solve(
                        GeometricHomotopy(
                            rotation_intrinsic_system,
                            start_parameters=offsetted_parameters,
                            target_parameters=parameters,
                        ),
                        monodromy_solutions;
                        show_progress=false
                    )
                end

                display("Found $(nsolutions(line_parametric_result)) solutions with line parametric homotopy")
                display("Of which $(nreal(line_parametric_result)) are real")

                if (nreal(line_parametric_result) == 0)
                    display("No solutions found with line parametric homotopy, skipping this instance.")
                    continue
                end

                solved_paths_line_parametric += nsolutions(line_parametric_result)
                tracked_paths_line_parametric += nsolutions(line_parametric_result) + nfailed(line_parametric_result)

                true_solution = true_solutions[i][1:(4+3*config["number_of_views"])]

                # Get the solution with the smallest distance to the true solution
                closest_solution_index = findmin([norm(true_solution - sol) for sol in real_solutions(line_parametric_result)])[2]
                closest_solution = real_solutions(line_parametric_result)[closest_solution_index]

                deltas = closest_solution - true_solution
                deltas = [abs(delta / true_value) for (delta, true_value) in zip(deltas, true_solution)]

                push!(delta_fx, deltas[1])
                push!(delta_fy, deltas[2])
                push!(delta_cx, deltas[3])
                push!(delta_cy, deltas[4])
                push!(delta_rotation, mean(deltas[5:(4+3*config["number_of_views"])]))
                push!(elapsed_time, t)

                display("Deltas for closest solution: $(deltas)")

                if any(sol -> isapprox(true_solution, sol; atol=1e-2), real_solutions(line_parametric_result))
                    solved_instances_line_parametric += 1
                end

            end

            display("Offset: $offset")
            display("Solved instances with line parametric homotopy: $solved_instances_line_parametric / $(length(scenes))")
            display("Solved paths with line parametric homotopy: $solved_paths_line_parametric / $tracked_paths_line_parametric")

            config["delta_fx"]= delta_fx
            config["delta_fy"]= delta_fy
            config["delta_cx"]= delta_cx
            config["delta_cy"]= delta_cy
            config["delta_rotation"]= delta_rotation
            config["elapsed_time"]= elapsed_time

            serialize("./tmp/scene_config_results.jls", cylinder_view_configurations)
        end

        serialize("./tmp/scene_config_results.jls", cylinder_view_configurations)
    end

    function csv_escape(value)
        value_str = string(value)
        if occursin('"', value_str)
            value_str = replace(value_str, '"' => "\"\"")
        end
        if occursin(',', value_str) || occursin('\n', value_str) || occursin('"', value_str)
            return "\"$(value_str)\""
        end
        return value_str
    end

    function export_scene_config_results(results_path="./tmp/scene_config_results.jls"; csv_output_path="./tmp/scene_config_results.csv")
        cylinder_view_configurations = deserialize(results_path)
        display(cylinder_view_configurations)

        print_scene_config_results(cylinder_view_configurations)

        header, data = scene_config_results_table_data(cylinder_view_configurations)
        open(csv_output_path, "w") do io
            println(io, join(header, ","))
            for i in axes(data, 1)
                row = [csv_escape(data[i, j]) for j in axes(data, 2)]
                println(io, join(row, ","))
            end
        end

        display("Scene configuration results exported to $(csv_output_path)")
    end

    function compare_parameter_homotopies()
        Random.seed!(777)
        intrinsic_configuration = IntrinsicParametersConfigurations.fₓ_fᵧ_cₓ_cᵧ

        scenes = []
        true_solutions = []

        for _ in 1:1
            intrinsics = [
                rand_in_range(2500.0, 2700.0) 0.0 rand_in_range(950.0, 970.0);   # fₓ, skew, cₓ
                0.0 rand_in_range(1400.0, 1600.0) rand_in_range(530.0, 550.0);   # 0, fᵧ, cᵧ
                0.0 0.0 1.0                                                      # bottom row
            ]

            cylinders = arbitrary_rig()

            cameras = []
            for _ in 1:2
                position, rotation = random_camera_lookingat_center()
                camera = CameraProperties()
                camera.position = position
                camera.quaternion_rotation = rotation
                camera.intrinsic = intrinsics
                push!(cameras, camera)
            end

            camera_view_pairs = [
                CameraViewPair(0, cameras[1], get_view(cylinders, cameras[1])),
                CameraViewPair(1, cameras[2], get_view(cylinders, cameras[2]))
            ]
            # map(((i, camera)) -> CameraViewPair(i - 1, camera, get_view(cylinders, camera)), enumerate(cameras))

            scene = SceneData()
            scene.figure = initfigure()
            scene.cylinders = cylinders
            scene.instances = map(camera_view_pair -> begin
                instance = InstanceConfiguration()
                instance.camera = camera_view_pair.camera
                instance.conics_contours = camera_view_pair.view
                return instance
            end, camera_view_pairs)

            push!(scenes, scene)
        end

        #region true solutions for original problems
        for scene in scenes
            true_intrinsics = scene.instances[1].camera.intrinsic ./ 3000.0
            rot1 = Rotations.params(QuatRotation(scene.instances[1].camera.rotation_matrix))
            rot1 = rot1 / rot1[1]
            rot1 = rot1[2:4]
            rot2 = Rotations.params(QuatRotation(scene.instances[2].camera.rotation_matrix))
            rot2 = rot2 / rot2[1]
            rot2 = rot2[2:4]

            true_solution = [
                true_intrinsics[1, 1],
                true_intrinsics[2, 2],
                true_intrinsics[1, 3],
                true_intrinsics[2, 3],
                rot1...,
                rot2...,
            ]
            push!(true_solutions, true_solution)
        end
        #endregion

        offset = 3.0
        offset_feasible = false
        while !offset_feasible
            display("Trying offset: $offset")

            solved_instances_parametric = 0
            solved_instances_line_parametric = 0
            solved_paths_parametric = 0
            solved_paths_line_parametric = 0
            tracked_paths_parametric = 0
            tracked_paths_line_parametric = 0

            for (i, scene) in enumerate(scenes)
                display("Processing scene: $i")
                problems = problems_from_scene(scene)
                _, parameters = intrinsic_rotation_system_setup(problems; intrinsic_configuration)

                offsetted_scene = SceneData()
                offsetted_scene.figure = initfigure()
                offsetted_scene.cylinders = scene.cylinders

                offsetted_intrinsics = scene.instances[1].camera.intrinsic + [
                    (rand(Float64) * 50.0 - 25.0) 0.0 (rand(Float64) * 10.0 - 5.0);   # fₓ, skew, cₓ
                    0.0 (rand(Float64) * 50.0 - 25.0) (rand(Float64) * 10.0 - 5.0);   # 0, fᵧ, cᵧ
                    0.0 0.0 0.0                                                      # bottom row
                ]

                offsetted_scene.instances = map((instance) -> begin
                    offsetted_camera = CameraProperties()
                    offsetted_camera.intrinsic = offsetted_intrinsics
                    random_direction = normalize(randn(3))
                    offsetted_camera.position = instance.camera.position + offset * random_direction
                    rotation = lookat_quaternion(offsetted_camera.position, [0.0, 0.0, 0.0])
                    offsetted_camera.quaternion_rotation = rotation
                    view = get_view(offsetted_scene.cylinders, offsetted_camera)
                    instance = InstanceConfiguration()
                    instance.camera = offsetted_camera
                    instance.conics_contours = view
                    return instance
                end, scene.instances)

                offsetted_problems = problems_from_scene(offsetted_scene)

                rotation_intrinsic_system, offsetted_parameters = intrinsic_rotation_system_setup(offsetted_problems;
                    intrinsic_configuration
                )

                #region Computing original sol
                rot = Rotations.params(QuatRotation(offsetted_scene.instances[1].camera.rotation_matrix))
                rot = rot / rot[1]
                rot1 = rot[2:4]
                rot = Rotations.params(QuatRotation(offsetted_scene.instances[2].camera.rotation_matrix))
                rot = rot / rot[1]
                rot2 = rot[2:4]
                known_solution = [
                    offsetted_scene.instances[1].camera.intrinsic[1, 1] / 3000.0,
                    offsetted_scene.instances[1].camera.intrinsic[2, 2] / 3000.0,
                    offsetted_scene.instances[1].camera.intrinsic[1, 3] / 3000.0,
                    offsetted_scene.instances[1].camera.intrinsic[2, 3] / 3000.0,
                    rot1...,
                    rot2...,
                ]
                #endregion

                monodromy_solutions = solutions(
                    monodromy_solve(
                        rotation_intrinsic_system,
                        [known_solution],
                        offsetted_parameters;
                        show_progress=false,
                    )
                )

                parametric_result = solve(
                    rotation_intrinsic_system,
                    monodromy_solutions;
                    start_parameters=offsetted_parameters,
                    target_parameters=parameters,
                    show_progress=false
                )

                display("Found $(nsolutions(parametric_result)) solutions with parametric homotopy")
                display("Of which $(nreal(parametric_result)) are real")

                line_parametric_result = solve(
                    GeometricHomotopy(
                        rotation_intrinsic_system,
                        start_parameters=offsetted_parameters,
                        target_parameters=parameters,
                    ),
                    monodromy_solutions;
                    show_progress=true
                )

                display("Known solution: $known_solution")
                display("Parametric")
                feasible_solutions = findall(sol -> isapprox(norm(known_solution - sol), 0.0; atol=1e-4), real_solutions(parametric_result))
                for rs in feasible_solutions
                    display(rs - known_solution)
                end
                display("Line parametric")
                feasible_solutions = findall(sol -> isapprox(norm(known_solution - sol), 0.0; atol=1e-4), real_solutions(line_parametric_result))
                for rs in feasible_solutions
                    display(rs - known_solution)
                end

                display("Found $(nsolutions(line_parametric_result)) solutions with line parametric homotopy")
                display("Of which $(nreal(line_parametric_result)) are real")

                solved_paths_parametric += nsolutions(parametric_result)
                tracked_paths_parametric += nsolutions(parametric_result) + nfailed(parametric_result)
                solved_paths_line_parametric += nsolutions(line_parametric_result)
                tracked_paths_line_parametric += nsolutions(line_parametric_result) + nfailed(line_parametric_result)

                if any(sol -> isapprox(true_solutions[i], sol; atol=1e-2), real_solutions(parametric_result))
                    solved_instances_parametric += 1
                end
                if any(sol -> isapprox(true_solutions[i], sol; atol=1e-2), real_solutions(line_parametric_result))
                    solved_instances_line_parametric += 1
                end
            end

            if solved_instances_parametric > 0 && solved_instances_line_parametric > 0
                offset_feasible = true

                display("Offset: $offset")
                display("Solved instances with parametric homotopy: $solved_instances_parametric / $(length(scenes))")
                display("Solved instances with line parametric homotopy: $solved_instances_line_parametric / $(length(scenes))")
                display("Solved paths with parametric homotopy: $solved_paths_parametric / $tracked_paths_parametric")
                display("Solved paths with line parametric homotopy: $solved_paths_line_parametric / $tracked_paths_line_parametric")
            else
                offset -= 0.5
            end

            display("--------------------------------------------------")
        end
    end

    function infinite_homography_homotopy()
        random_seed = 84564
        Random.seed!(random_seed)

        # 4 vanishing points needed to compute H_∞ via DLT (4 correspondences)
        # But the homotopy system uses only 2 lines (that intersect at a point)
        n_vps_for_hinf = 4
        n_lines_for_system = 2

        # 4 random 3D vanishing points (directions)
        vanishing_points_3d = [normalize(randn(3)) for _ in 1:n_vps_for_hinf]
        display("3D Vanishing points (directions) for H_∞ computation:")
        for (i, vp) in enumerate(vanishing_points_3d)
            display("  VP $i: $vp")
        end

        # Create 2 random cameras with shared intrinsics
        intrinsics = [
            rand_in_range(2500.0, 2700.0) 0.0 rand_in_range(950.0, 970.0);   # fₓ, skew, cₓ
            0.0 rand_in_range(1400.0, 1600.0) rand_in_range(530.0, 550.0);   # 0, fᵧ, cᵧ
            0.0 0.0 1.0                                                      # bottom row
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

        display("Camera 1 position: $(cameras[1].position)")
        display("Camera 2 position: $(cameras[2].position)")

        # Project 3D points to each view (homogeneous coordinates)
        function project_point(camera, pt_3d)
            pt_4d = [pt_3d; 1.0]
            projected = camera.matrix * pt_4d
            return projected  # Keep as homogeneous 3-vector
        end

        # Get all 4 vanishing points in each view (for H_∞ computation)
        all_vps_view1 = [project_point(cameras[1], d) for d in vanishing_points_3d]
        all_vps_view2 = [project_point(cameras[2], d) for d in vanishing_points_3d]

        display("All vanishing points in view 1 (for H_∞):")
        for (i, vp) in enumerate(all_vps_view1)
            display("  VP $i: $vp")
        end

        # Compute H_∞ using DLT from all 4 vanishing point correspondences
        pts1 = vcat([v[1:2]' ./ v[3] for v in all_vps_view1]...)  # N x 2 (inhomogeneous)
        pts2 = vcat([v[1:2]' ./ v[3] for v in all_vps_view2]...)  # N x 2 (inhomogeneous)
        Hinf = compute_Hinf(pts1, pts2)
        display("H_∞ (computed via DLT from 4 correspondences):")
        display(Hinf)

        # Verify H_∞ maps vanishing points correctly
        display("Verification: H_∞ maps VP1 -> VP2:")
        for i in 1:n_vps_for_hinf
            vp2_pred = Hinf * all_vps_view1[i]
            vp2_pred = normalize(vp2_pred)
            similarity = abs(dot(vp2_pred, all_vps_view2[i]))
            display("  VP $i: similarity = $similarity (should be ≈1)")
        end

        # Select 2 vanishing points for the system (the lines that will intersect)
        # Use first 2 vanishing points
        vps_view1 = all_vps_view1[1:n_lines_for_system]
        vps_view2 = all_vps_view2[1:n_lines_for_system]

        display("Vanishing points for system (2 lines):")
        for (i, vp) in enumerate(vps_view1)
            display("  VP $i view1: $vp")
            display("  VP $i view2: $(vps_view2[i])")
        end

        # Create lines by projecting real 3D points
        # For each line: create a random 3D point, project to both views,
        # then the line passes through the VP and the projected point
        points_3d = [randn(3) for _ in 1:n_lines_for_system]
        display("Random 3D points for lines:")
        for (i, pt) in enumerate(points_3d)
            display("  Point $i: $pt")
        end

        points_view1 = [project_point(cameras[1], pt) for pt in points_3d]
        points_view2 = [project_point(cameras[2], pt) for pt in points_3d]

        display("Projected points in view1:")
        for (i, pt) in enumerate(points_view1)
            pt_inhom = pt[1:2] ./ pt[3]
            display("  Point $i: $pt_inhom")
        end
        display("Projected points in view2:")
        for (i, pt) in enumerate(points_view2)
            pt_inhom = pt[1:2] ./ pt[3]
            display("  Point $i: $pt_inhom")
        end

        # Create lines: line through VP and projected point (cross product of two points = line)
        lines_view1 = [normalize(cross(vps_view1[i], points_view1[i])) for i in 1:n_lines_for_system]
        lines_view2 = [normalize(cross(vps_view2[i], points_view2[i])) for i in 1:n_lines_for_system]

        display("Lines for system (2 per view, through VP and projected 3D point):")
        for (i, l) in enumerate(lines_view1)
            incidence_vp = dot(l, vps_view1[i])
            incidence_pt = dot(l, points_view1[i])
            display("  Line $i view1: $l (incidence VP: $incidence_vp, incidence point: $incidence_pt)")
        end
        for (i, l) in enumerate(lines_view2)
            incidence_vp = dot(l, vps_view2[i])
            incidence_pt = dot(l, points_view2[i])
            display("  Line $i view2: $l (incidence VP: $incidence_vp, incidence point: $incidence_pt)")
        end

        # Compute intersection point of the 2 lines in each view
        intersection_view1 = cross(lines_view1[1], lines_view1[2])
        intersection_view1 = intersection_view1 ./ intersection_view1[3]
        intersection_view2 = cross(lines_view2[1], lines_view2[2])
        intersection_view2 = intersection_view2 ./ intersection_view2[3]

        display("Intersection of 2 lines in view1: $(intersection_view1[1:2])")
        display("Intersection of 2 lines in view2: $(intersection_view2[1:2])")

        # Flatten lines to parameter vectors (2 lines = 6 parameters)
        p = vcat(lines_view1...)  # start parameters (view 1)
        q = vcat(lines_view2...)  # target parameters (view 2)

        display("Start parameters (2 lines from view 1): $p")
        display("Target parameters (2 lines from view 2): $q")

        # Create a polynomial system with 2 equations (point lies on both lines)
        # Variables: x, y (2D point coordinates)
        # Equations: l1·[x,y,1] = 0, l2·[x,y,1] = 0
        @var x y
        @var l1[1:3] l2[1:3]
        eq1 = l1[1]*x + l1[2]*y + l1[3]
        eq2 = l2[1]*x + l2[2]*y + l2[3]
        F = System([eq1, eq2]; variables=[x, y], parameters=[l1; l2])

        display("Polynomial system (2 lines intersecting):")
        display(F)

        # Create the InfiniteHomographyHomotopy
        display("Creating InfiniteHomographyHomotopy...")
        homotopy = InfiniteHomographyHomotopy(
            F,
            p,
            q,
            Hinf,
            vps_view1  # 2 vanishing points for the 2 lines
        )

        display("XXXXXXXXXXXXXXXXXXXXXXXXXX")
        display(F.expressions)
        display(F.parameters)
        display(subs(expressions(F), l1 => lines_view2[1], l2 => lines_view2[2]))
        display(lines_view2)
        display(lines_view2[1])
        display(lines_view2[2])
        display("XXXXXXXXXXXXXXXXXXXXXXXXXX")


        result = solve(
            homotopy,
            [
                intersection_view1[1:2],
            ];
            show_progress=true
        )

        sols = real_solutions(result)
        display("Found $(nsolutions(result)) solutions for the intersection point:")

        for (i, sol) in enumerate(sols)
            display("  Solution $i: $(sol)")
            # Verify that the solution lies on both lines in view 2
            l1 = lines_view2[1]
            l2 = lines_view2[2]
            incidence1 = abs(dot(l1, [sol; 1.0]))
            incidence2 = abs(dot(l2, [sol; 1.0]))
            display("    Incidence with line 1 in view 2: $incidence1 (should be ≈0)")
            display("    Incidence with line 2 in view 2: $incidence2 (should be ≈0)")
        end

        display(dot(lines_view2[1], intersection_view2))
        display(dot(lines_view2[2], intersection_view2))
        display(intersection_view2)

        return sols

        # # Test line interpolation at various t values
        # t_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        # display("Testing line interpolation:")

        # for t in t_values
        #     display("  t = $t:")
        #     for i in 1:n_lines_for_system
        #         # Get interpolated line
        #         line_t = Homotopies.interpolate_line(homotopy, i, t)

        #         # Compute interpolated vanishing point: v_t = H_t * v_0
        #         H_t = Homotopies.matrix_exp(t * homotopy.log_H_inf)
        #         v_t = normalize(H_t * vps_view1[i])

        #         # Check incidence: ℓ_tᵀ * v_t should be ≈ 0
        #         incidence = abs(dot(line_t, v_t))
        #         display("    Line $i: incidence error = $incidence")

        #         # Verify boundary conditions
        #         if t == 0.0
        #             similarity = abs(dot(line_t, normalize(lines_view1[i])))
        #             display("    Line $i: similarity to start = $similarity")
        #         elseif t == 1.0
        #             similarity = abs(dot(line_t, normalize(lines_view2[i])))
        #             display("    Line $i: similarity to target = $similarity")
        #         end
        #     end

        #     # Show the intersection point at this t
        #     Homotopies.tp!(homotopy, t)
        #     line1_t = real.(homotopy.pt[1:3])
        #     line2_t = real.(homotopy.pt[4:6])
        #     intersection_t = cross(line1_t, line2_t)
        #     intersection_t = intersection_t ./ intersection_t[3]
        #     display("    Intersection point at t=$t: $(intersection_t[1:2])")
        # end

        # # Solve the system at t=0 and t=1 to verify
        # display("Solving system at boundaries:")
        # solution_start = intersection_view1[1:2]
        # solution_target = intersection_view2[1:2]
        # display("  Solution at t=0 (view1 intersection): $solution_start")
        # display("  Solution at t=1 (view2 intersection): $solution_target")

        # return homotopy, cameras, vps_view1, vps_view2, lines_view1, lines_view2, Hinf
    end
end
