module CylindersBasedCameraResectioning
    const GUI_ENABLED = get(ENV, "GUI_ENABLED", "true") == "true"
    display("GUI_ENABLED: $GUI_ENABLED")
    const ASSERTS_ENABLED = get(ENV, "ASSERTS_ENABLED", "false") == "true"
    const IMAGE_HEIGHT = 1080
    const IMAGE_WIDTH = 1920
    include("includes.jl")

    using ..Cylinder: cylinder_from_center_axis_radius, points_at_infinity_dualquadrics
	using ..Scene: SceneData, InstanceConfiguration, ParametersSolutionsPair, averaged_solution!, best_overall_solution!, best_overall_solution_by_steps!, best_intrinsic_rotation_translation_system_solution!, camera_from_solution, create_scene_instances_and_problems, scene_instances_and_problems_from_files, intrinsic_rotation_system_setup, intrinsic_rotation_translation_system_setup, plot_interactive_scene, plot_reconstructed_scene, split_intrinsic_rotation_parameters, plot_scene, best_intrinsic_rotation_system_solution!, add_noise_to_line
    using ..Camera: CameraProperties, CameraViewPair, lookat_quaternion
    using ..Geometry: get_view
	using ..EquationSystems: stack_homotopy_parameters, variables_jacobian_rank, joint_jacobian_rank
    using ..EquationSystems.Problems: CylinderCameraContoursProblem, CylinderCameraContoursProblemValidationData
    using ..EquationSystems.Problems.IntrinsicParameters: Configurations as IntrinsicParametersConfigurations
    using ..Plotting
	using ..Printing: print_camera_differences, print_relative_motion_errors, save_results_to_json
    using ..Homotopies: ParameterHomotopy as MyParameterHomotopy
    using ..IO: read_point_cloud_zup, read_cylinders_zup, read_camera_from_matrices, read_cylinder_line_views
    using ..Utils: lines_clp_to_stack
    using ..Homotopies: GeometricHomotopy

    using HomotopyContinuation, Observables, Random, Serialization, LinearAlgebra
    using Combinatorics: permutations as perms

    function main()
        intrinsic_configuration = IntrinsicParametersConfigurations.none
        scene, problems = create_scene_instances_and_problems(;
            number_of_instances=1,
            number_of_cylinders=2,
            random_seed=1243,
            intrinsic_configuration,
            noise=0.0,
        )

        display(scene.figure)

        rotation_intrinsic_system, parameters = intrinsic_rotation_system_setup(
            problems
        )
        display(rotation_intrinsic_system)
        display(parameters)

        start_solutions = nothing
        start_parameters = nothing
        # try
        #     parameters_solutions_pair = deserialize("tmp/intrinsic_rotation_monodromy_solutions.jld")
        #     start_solutions = parameters_solutions_pair.solutions
        #     start_parameters = parameters_solutions_pair.start_parameters
        # catch e
        #     @error e
        #     display("No intrinsic-rotation monodromy")
        # end

        solver = starts = nothing

        if isnothing(start_parameters)
            solver, starts = solver_startsolutions(
                rotation_intrinsic_system;
                target_parameters = parameters,
                start_system = :total_degree,
            )
        else
            display("Using parameter homotopy")
            geometric_homotopy = MyParameterHomotopy(rotation_intrinsic_system, start_parameters, parameters)
            solver, starts = solver_startsolutions(
                geometric_homotopy,
                start_solutions;
            )
        end

        chunk_size = 500000
        numberof_start_solutions = length(starts)
        display("Number of start solutions: $numberof_start_solutions. Number of iterations needed: $(ceil(Int, numberof_start_solutions / chunk_size))")
        start_error = Inf
        for start in Iterators.partition(starts, chunk_size)
            result = solve(
                solver,
                start;
            )
            # @info result

            start_error, _ = best_overall_solution_by_steps!(
                result,
                problems;
                intrinsic_configuration,
                start_error=start_error,
                # scene,
            )
        end

        for (i, instance) in enumerate(scene.instances)
            display("View $i")
            print_camera_differences(instance.camera, problems[i].camera)
            display("--------------------")
        end

        for problem in problems
            plot_3dcamera(Plot3dCameraInput(
                problem.camera.euler_rotation,
                problem.camera.position,
            ), :green)
        end

        plot_reconstructed_scene(scene, problems)

        display(scene.figure)
    end

    function monodromy()
        paramters_solutions_pairs::Vector{ParametersSolutionsPair} = []
        Random.seed!(67)

        for try_index in 1:1000
            try
                intrinsic_rotations_monodromy_solutions::Vector{Vector{ComplexF64}} = []
                translations_monodromy_solutions::Vector{Vector{ComplexF64}} = []

                camera_random_seed = rand(Int)
                scene, problems = create_scene_instances_and_problems(;
                    number_of_instances=2,
                    number_of_cylinders=3,
                    random_seed=camera_random_seed,
                    cylinders_random_seed=14,
                    intrinsic_configuration=IntrinsicParametersConfigurations.fₓ_fᵧ_cₓ_cᵧ,
                )

                rotation_intrinsic_system, parameters = intrinsic_rotation_system_setup(problems)
                # _, total_degree_solutions = total_degree(rotation_intrinsic_system; target_parameters = parameters)

                fₓ, _, _, skew, fᵧ, _, cₓ, cᵧ, _ = vec(scene.instances[1].camera.intrinsic)
                startingsolution = [fₓ, fᵧ, cₓ, cᵧ]

                for instance in scene.instances
                    camera = instance.camera
                    rot = Rotations.params(camera.quaternion_rotation')
                    rot = rot / rot[1]
                    rot = rot[2:end]
                    startingsolution = stack_homotopy_parameters(
                        startingsolution,
                        rot,
                    )
                end

                @info "Jacobians rank"
                @info variables_jacobian_rank(rotation_intrinsic_system, startingsolution, parameters)
                @info joint_jacobian_rank(rotation_intrinsic_system, startingsolution, parameters)

                startingsolution = convert(Vector{Float64}, startingsolution)
                # parameters = convert(Vector{Float64}, parameters)

                push!(intrinsic_rotations_monodromy_solutions, startingsolution)

                while true
                    old_solution_count = length(intrinsic_rotations_monodromy_solutions)
                    monodromy_solutions = monodromy_solve(
                        rotation_intrinsic_system,
                        intrinsic_rotations_monodromy_solutions,
                        parameters;
                    )
                    @info monodromy_solutions
                    push!(intrinsic_rotations_monodromy_solutions, solutions(monodromy_solutions)...)
                    intrinsic_rotations_monodromy_solutions = unique(intrinsic_rotations_monodromy_solutions)
                    new_solution_count = length(intrinsic_rotations_monodromy_solutions)

                    try
                        mkdir("tmp")
                    catch
                    end

                    # for (i, problem) in enumerate(problems)
                    #     problem.camera.intrinsic = scene.instances[i].camera.intrinsic
                    #     problem.camera.quaternion_rotation = scene.instances[i].camera.quaternion_rotation
                    #     translation_system, parameters = intrinsic_rotation_translation_system_setup(problem)
                    #     startingsolution = scene.instances[i].camera.position
                    #     startingsolution = convert(Vector{Float64}, startingsolution)
                    #     monodromy_solutions = monodromy_solve(
                    #         translation_system,
                    #         startingsolution,
                    #         parameters
                    #     )
                    #     @info monodromy_solutions
                    #     push!(translations_monodromy_solutions, solutions(monodromy_solutions)...)
                    #     translations_monodromy_solutions = unique(translations_monodromy_solutions)
                    # end

                    if new_solution_count == old_solution_count
                        break
                    end
                end

                for sol in intrinsic_rotations_monodromy_solutions
                    e = evaluate(rotation_intrinsic_system, sol, parameters)
                    tot_error = sum(abs.(e))
                    if tot_error > 1e-6
                        @error "Error: $tot_error"
                    end
                end

                # append!(intrinsic_rotations_monodromy_solutions, collect(total_degree_solutions))

                push!(paramters_solutions_pairs, ParametersSolutionsPair(
                    parameters,
                    intrinsic_rotations_monodromy_solutions
                ))
            catch e
                @error e
                display("No intrinsic-rotation monodromy")
            end
        end

        serialize(
            "tmp/intrinsic_rotation_monodromy_solutions.jld",
            paramters_solutions_pairs
        )

        # serialize(
        #     "tmp/translation_monodromy_solutions.jld",
        #     ParametersSolutionsPair(
        #         parameters,
        #         translations_monodromy_solutions
        #     )
        # )
    end

    function explore_path()
        intrinsic_configuration = IntrinsicParametersConfigurations.fₓ_fᵧ
        scene_target, problems_target = create_scene_instances_and_problems(;
            number_of_instances=2,
            number_of_cylinders=2,
            random_seed=14,
            intrinsic_configuration,
            plot=false,
        )
        scene_start, problems_start = create_scene_instances_and_problems(;
            number_of_instances=2,
            number_of_cylinders=2,
            random_seed=67,
            cylinders_random_seed=14,
            intrinsic_configuration,
            plot=false,
        )

        rotation_intrinsic_system_start, parameters_start = intrinsic_rotation_system_setup(problems_start)
        rotation_intrinsic_system_target, parameters_target = intrinsic_rotation_system_setup(problems_target)
        homotopy = ParameterHomotopy(rotation_intrinsic_system_start, parameters_start, parameters_target)
        tracker = Tracker(homotopy)

        fₓ, _, _, skew, fᵧ, _, cₓ, cᵧ, _ = vec(scene_start.instances[1].camera.intrinsic)
        startingsolution = [fₓ, fᵧ]
        for instance in scene_start.instances
            camera = instance.camera
            rot = Rotations.params(camera.quaternion_rotation')
            rot = rot / rot[1]
            rot = rot[2:end]
            startingsolution = stack_homotopy_parameters(
                startingsolution,
                rot,
            )
        end

        pts = []
        for (x, t) in iterator(tracker, startingsolution, 1.0, 0.0)
            push!(pts, real.(x))
        end

        current_figure = initfigure()
        slider = add_slider!(;
            start=1,
            stop=length(pts),
            step=1,
        )
        for i in 1:length(scene_start.instances)
            add_camera_rotation_axis!()
        end
        observable_instances = lift(slider.value) do solution_index
            instances = []
            solution = pts[solution_index]
            intrinsic = rotations_solution = intrinsic_correction = nothing
            try
                intrinsic, rotations_solution, intrinsic_correction = split_intrinsic_rotation_parameters(
                    solution,
                    intrinsic_configuration
                )
            catch e
                @error e
            end
            for (i, instance) in enumerate(scene_start.instances)
                push!(instances, deepcopy(instance))
                if solution_index != 1
                    camera = camera_from_solution(
                        intrinsic,
                        rotations_solution,
                        intrinsic_correction,
                        i
                    )
                    problem = deepcopy(problems_start[i])
                    problem.camera = camera
                    # translation_system, parameters = intrinsic_rotation_translation_system_setup(problem)

                    # translation_result = solve(
                    #     translation_system,
                    #     target_parameters = parameters,
                    #     start_system = :total_degree,
                    # )
                    # @info translation_result

                    # best_intrinsic_rotation_translation_system_solution!(
                    #     translation_result,
                    #     scene_start,
                    #     instances[i],
                    #     problem
                    # )
                    instances[i].camera = problem.camera
                end
            end
            instances
        end

        plot_interactive_scene(;
            scene=scene_target,
            problems=problems_start,
            observable_instances,
            figure=current_figure,
        )
        display(current_figure)
    end

    function markers()
        scene, problems = scene_instances_and_problems_from_files(
            "./assets/test_scenes/markers/scene.json",
            "./assets/methods_compare/real/markers.json";
            number_of_instances=2,
            use_all_lines=true
        )
        intrinsic_configuration = problems[1].intrinsic_configuration

        noise = 1.0
        for (i, instance) in enumerate(scene.instances)
            instance.conics_contours = get_view(scene.cylinders, instance.camera)
            for j in 1:size(instance.conics_contours, 1)
                instance.conics_contours[j, 1, :] = add_noise_to_line(instance.conics_contours[j, 1, :], noise)
                instance.conics_contours[j, 2, :] = add_noise_to_line(instance.conics_contours[j, 2, :], noise)
                instance.conics_contours[j, 1, :] /= instance.conics_contours[j, 1, 3]
                instance.conics_contours[j, 2, :] /= instance.conics_contours[j, 2, 3]
                display("View $i, cylinder $j")
                display(instance.conics_contours[j, :, :])
            end
            problems[i].lines = lines_clp_to_stack(instance.conics_contours)
        end

        # save_2d_figures("assets/test_scenes/markers/figures/", scene, problems; prefix = "initial_", scene_file_path = "./assets/test_scenes/markers/scene.json")

        equation_combinations = [1, 3, 5, 7, 9, 2, 4, 6, 8, 10]

        rotation_intrinsic_system, parameters = intrinsic_rotation_system_setup(problems;
            intrinsic_configuration,
            equation_combinations,
            minimization=false,
        )

    #    result = solve(
    #         rotation_intrinsic_system;
    #         target_parameters = parameters,
    #         start_system = :total_degree,
    #     )

    #     serialize(
    #         "tmp/markers_result.jld",
    #         result
    #     )

        result = deserialize("tmp/markers_result.jld")
        best_overall_solution_by_steps!(
            result,
            problems;
            start_error=Inf,
            intrinsic_configuration,
            scene,
        )

        for (i, instance) in enumerate(scene.instances)
            display("View $i")
            print_camera_differences(instance.camera, problems[i].camera)
            display("--------------------")
        end

        print_relative_motion_errors(scene, problems)
    end

    function pipes()
        scene, problems = scene_instances_and_problems_from_files(
            "./assets/test_scenes/pipes/scene.json",
            "./assets/test_scenes/pipes/views.json";
            number_of_instances=2,
        )
        intrinsic_configuration = problems[1].intrinsic_configuration

        rotation_intrinsic_system, parameters = intrinsic_rotation_system_setup(problems)

        display(scene.figure)

        solver = starts = nothing

        solver, starts = solver_startsolutions(
            rotation_intrinsic_system;
            target_parameters = parameters,
            start_system = :total_degree,
        )

        chunk_size = 500000
        numberof_start_solutions = length(starts)
        display("Number of start solutions: $numberof_start_solutions. Number of iterations needed: $(ceil(Int, numberof_start_solutions / chunk_size))")
        solution_error = Inf
        for start in Iterators.partition(starts, chunk_size)
            result = solve(
                solver,
                start;
            )
            @info result

            solution_error, _ = best_overall_solution_by_steps!(
                result,
                problems;
                start_error=solution_error,
                intrinsic_configuration,
                validation_cylinders = scene.cylinders,
            )
        end
        solution_error = solution_error / 2
        display("Solution error: $solution_error")

        for (i, instance) in enumerate(scene.instances)
            display("View $i")
            print_camera_differences(instance.camera, problems[i].camera)
            display("--------------------")
        end

        # print_relative_motion_errors(scene, problems)

        for problem in problems
            plot_3dcamera(problem.camera, :green)
        end

        plot_reconstructed_scene(scene, problems)

        save_2d_figures("assets/test_scenes/pipes/figures/", scene, problems; prefix = "localization", scene_file_path = "./assets/test_scenes/pipes/scene.json")

        display(scene.figure)
    end

    function pipes_animation()
        previous_solutions = nothing
        previous_parameters = nothing
        for i in 1:12
            display("Frame $i")
            offset = i-1

            scene_file_path = "./assets/test_scenes/pipes/animation/scene.json"

            scene, problems = scene_instances_and_problems_from_files(
                scene_file_path,
                "./assets/test_scenes/pipes/animation/views.json";
                number_of_instances=1,
                instance_offset = offset,
            )
            intrinsic_configuration = problems[1].intrinsic_configuration

            rotation_intrinsic_system, parameters = intrinsic_rotation_system_setup(problems)

            result = nothing

            if isnothing(previous_solutions)
                result = solve(
                    rotation_intrinsic_system;
                    target_parameters=parameters,
                    start_system = :total_degree,
                )
                @info result
            else
                result = solve(
                    rotation_intrinsic_system,
                    previous_solutions,
                    start_parameters=previous_parameters,
                    target_parameters=parameters,
                )
                @info result
            end

            _, _, best_solution = best_overall_solution_by_steps!(
                result,
                problems;
                intrinsic_configuration,
            )

            # best_rotation_system_solution = best_solution[1:end-3]
            # display("Monodromy solutions for view $i")
            # monodromy_solutions = monodromy_solve(
            #     rotation_intrinsic_system,
            #     best_rotation_system_solution,
            #     parameters;
            # )
            # display([best_rotation_system_solution; solutions(monodromy_solutions)...])
            # previous_solutions = [best_rotation_system_solution; solutions(monodromy_solutions)...]
            previous_solutions = solutions(result)
            previous_parameters = parameters

            try
                save_2d_figures("assets/test_scenes/pipes/animation/figures/", scene, problems; scene_file_path, prefix="view_$(i)_", image_offset=offset)
            catch e
                Base.showerror(stdout, e)
				Base.show_backtrace(stdout, catch_backtrace())
            end
        end
    end

    function roller_coaster()
        scene, problems = scene_instances_and_problems_from_files(
            "./assets/test_scenes/roller_coaster/scene.json",
            "./assets/test_scenes/roller_coaster/views.json";
            number_of_instances=1,
            cylinders_names_in_view_file=[
                "cylinder_0",
                "cylinder_1",
                "cylinder_2",
                "cylinder_3",
                "cylinder_4",
            ],
            plot_3d = false,
        )
        intrinsic_configuration = problems[1].intrinsic_configuration

        rotation_intrinsic_system, parameters = intrinsic_rotation_system_setup(problems;
            intrinsic_configuration,
            minimization=true,
        )

        display(scene.figure)

        solver = starts = nothing

        solver, starts = solver_startsolutions(
            rotation_intrinsic_system;
            target_parameters = parameters,
            start_system = :total_degree,
        )

        chunk_size = 500000
        numberof_start_solutions = length(starts)
        display("Number of start solutions: $numberof_start_solutions. Number of iterations needed: $(ceil(Int, numberof_start_solutions / chunk_size))")
        solution_error = Inf
        for start in Iterators.partition(starts, chunk_size)
            result = solve(
                solver,
                start;
            )
            @info result

            solution_error, _ = best_overall_solution_by_steps!(
                result,
                problems;
                start_error=solution_error,
                intrinsic_configuration,
                # scene,
            )
        end

        display("Solution error: $solution_error")

        for (i, instance) in enumerate(scene.instances)
            display("View $i")
            print_camera_differences(instance.camera, problems[i].camera)
            display("--------------------")
        end

        print_relative_motion_errors(scene, problems)

        plot_reconstructed_scene(scene, problems; plot_3d = false, raw_3d = true)

        save_2d_figures("assets/test_scenes/roller_coaster/figures/", scene, problems; scene_file_path = "./assets/test_scenes/roller_coaster/scene.json", raw_3d = true)

        display(scene.figure)
    end

    function water_tower()
        intrinsic_configuration = IntrinsicParametersConfigurations.fₓ_fᵧ_cₓ_cᵧ
        filepath = "./assets/test_scenes/water_tower/scene.json"
        cylinders = read_cylinders_zup(filepath)
        cylinders = [cylinder_from_center_axis_radius(cylinder.center, cylinder.axis, cylinder.radius)
            for cylinder in cylinders]
        camera1 = read_camera_from_matrices(
            filepath,
            1
        )
        camera2 = read_camera_from_matrices(
            filepath,
            79
        )
        views = read_cylinder_line_views(filepath; object_path="lines")

        camera_view_pairs = [
            CameraViewPair(1, camera1, views[1]),
            CameraViewPair(2, camera2, views[2]),
        ]

        scene = SceneData()
        scene.figure = initfigure()
        scene.cylinders = cylinders

        scene.instances = map(camera_view_pair -> begin
                instance = InstanceConfiguration()
                instance.camera = camera_view_pair.camera
                instance.conics_contours = camera_view_pair.view
                return instance
            end, camera_view_pairs)

        points_at_infinity, dualquadrics = points_at_infinity_dualquadrics(cylinders)

        problems::Vector{CylinderCameraContoursProblem} = []
        validation_data = CylinderCameraContoursProblemValidationData(
            Matrix{Float64}(undef, 0, 3),
            Matrix{Float64}(undef, 0, 3),
            Array{Float64}(undef, 0, 4, 4),
        )
        for (i, camera_view_pair) in enumerate(camera_view_pairs)
            lines = lines_clp_to_stack(camera_view_pair.view)

            problem = CylinderCameraContoursProblem(
                camera_view_pair.camera,
                lines,
                lines,
                points_at_infinity,
                dualquadrics,
                validation_data,
                UInt8(intrinsic_configuration)
            )
            push!(problems, problem)
        end

        plot_scene(scene, problems)

        rotation_intrinsic_system, parameters = intrinsic_rotation_system_setup(
            problems;
            minimization=false,
            intrinsic_configuration,
        )

        result = solve(
            rotation_intrinsic_system;
            target_parameters=parameters,
            start_system=:total_degree,
        )

        @info result

        solution_error, _ = best_overall_solution_by_steps!(
            result,
            problems;
            intrinsic_configuration
        )

        display("Solution error: $solution_error")

        for (i, instance) in enumerate(scene.instances)
            display("View $i")
            print_camera_differences(instance.camera, problems[i].camera)
            display("--------------------")
        end

        print_relative_motion_errors(scene, problems)

        plot_reconstructed_scene(scene, problems; plot_3d = false, raw_3d = true)

        save_2d_figures("assets/test_scenes/water_tower/figures/", scene, problems; scene_file_path = "./assets/test_scenes/water_tower/scene.json", raw_3d = true)

        display(scene.figure)
    end

    function hot_pipes()
        intrinsic_configuration = IntrinsicParametersConfigurations.fₓ_fᵧ_cₓ_cᵧ
        filepath = "./assets/test_scenes/hot_pipes/scene.json"
        cylinders = read_cylinders_zup(filepath)
        cylinders = [cylinder_from_center_axis_radius(cylinder.center, cylinder.axis, cylinder.radius)
            for cylinder in cylinders]
        camera1 = read_camera_from_matrices(
            filepath,
            1
        )
        camera2 = read_camera_from_matrices(
            filepath,
            2
        )
        # views = read_cylinder_line_views(filepath; object_path="lines")

        views = map(camera -> begin
            get_view(cylinders, camera)
        end, [camera1, camera2])

        # noise = 0.01

        # for view in views
        #     for j in 1:size(view, 1)
        #         view[j, 1, :] = add_noise_to_line(view[j, 1, :], noise)
        #         view[j, 2, :] = add_noise_to_line(view[j, 2, :], noise)
        #         view[j, 1, :] /= view[j, 1, 3]
        #         view[j, 2, :] /= view[j, 2, 3]
        #     end
        # end

        camera_view_pairs = [
            CameraViewPair(1, camera1, views[1]),
            CameraViewPair(2, camera2, views[2]),
        ]

        scene = SceneData()
        scene.figure = initfigure()
        scene.cylinders = cylinders

        scene.instances = map(camera_view_pair -> begin
                instance = InstanceConfiguration()
                instance.camera = camera_view_pair.camera
                instance.conics_contours = camera_view_pair.view
                return instance
            end, camera_view_pairs)

        points_at_infinity, dualquadrics = points_at_infinity_dualquadrics(cylinders)

        problems::Vector{CylinderCameraContoursProblem} = []
        validation_data = CylinderCameraContoursProblemValidationData(
            Matrix{Float64}(undef, 0, 3),
            Matrix{Float64}(undef, 0, 3),
            Array{Float64}(undef, 0, 4, 4),
        )
        for (i, camera_view_pair) in enumerate(camera_view_pairs)
            lines = lines_clp_to_stack(camera_view_pair.view)

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

        trials = 0
        result = nothing
        while trials < 100
            display("Trial $trials")
            trials += 1
            init_scene = SceneData()
            init_scene.figure = initfigure()
            init_scene.cylinders = cylinders

            intrinsic_init = [
                2500.0 0.0 961.2;
                0.0 2100.0 538.0;
                0.0 0.0 1.0;
            ]
            random_direction = randn(3)
            random_direction /= norm(random_direction)
            position = 20.0 * random_direction
            rotation = lookat_quaternion(position, [0.0, 0.0, 0.0])

            camera1 = CameraProperties()
            camera1.intrinsic = intrinsic_init
            camera1.position = position
            camera1.quaternion_rotation = rotation

            position = position + [0.0, -30.0, 0.0]
            rotation = lookat_quaternion(position, [0.0, 0.0, 0.0])
            camera2 = CameraProperties()
            camera2.intrinsic = intrinsic_init
            camera2.position = position
            camera2.quaternion_rotation = rotation

            init_camera_view_pairs = [
                CameraViewPair(1, camera1, get_view(cylinders, camera1)),
                CameraViewPair(2, camera2, get_view(cylinders, camera2)),
            ]

            init_instances = map(camera_view_pair -> begin
                instance = InstanceConfiguration()
                instance.camera = camera_view_pair.camera
                instance.conics_contours = camera_view_pair.view
                return instance
            end, init_camera_view_pairs)

            init_scene.instances = init_instances

            init_camera_view_pairs = [
                CameraViewPair(1, camera1, get_view(cylinders, camera1)),
                CameraViewPair(2, camera2, get_view(cylinders, camera2)),
            ]

            init_problems::Vector{CylinderCameraContoursProblem} = []
            validation_data = CylinderCameraContoursProblemValidationData(
                Matrix{Float64}(undef, 0, 3),
                Matrix{Float64}(undef, 0, 3),
                Array{Float64}(undef, 0, 4, 4),
            )
            for (i, camera_view_pair) in enumerate(init_camera_view_pairs)
                lines = lines_clp_to_stack(camera_view_pair.view)

                problem = CylinderCameraContoursProblem(
                    CameraProperties(),
                    lines,
                    lines,
                    points_at_infinity,
                    dualquadrics,
                    validation_data,
                    UInt8(intrinsic_configuration)
                )
                push!(init_problems, problem)
            end

            #region Computing original sol
            rotations = map(instance -> begin
                rot = Rotations.params(QuatRotation(instance.camera.rotation_matrix))
                rot = rot / rot[1]
                rot = rot[2:4]
                return rot
            end, init_scene.instances)

            # Merge all rotations into a single vector
            merged_rotations = vcat(rotations...)
            known_solution = [
                init_scene.instances[1].camera.intrinsic[1, 1] / 3000.0,
                init_scene.instances[1].camera.intrinsic[2, 2] / 3000.0,
                init_scene.instances[1].camera.intrinsic[1, 3] / 3000.0,
                init_scene.instances[1].camera.intrinsic[2, 3] / 3000.0,
                merged_rotations...,
            ]
            #endregion

            rotation_intrinsic_system = target_parameters = start_parameters = nothing
            
            combinations_iterator = perms([(2:6)..., (7:11)...], 10)
            for equation_combinations in combinations_iterator
                _rotation_intrinsic_system, _target_parameters = intrinsic_rotation_system_setup(
                    problems;
                    intrinsic_configuration,
                    minimization=true,
                    equation_combinations,
                )
                _, _start_parameters = intrinsic_rotation_system_setup(
                    init_problems;
                    intrinsic_configuration,
                    minimization=true,
                    equation_combinations,
                )

                e = evaluate(_rotation_intrinsic_system, known_solution, _start_parameters)
                r = rank(jacobian(_rotation_intrinsic_system, known_solution, _start_parameters))

                display("Trying combination: $equation_combinations, error: $(norm(e)), rank: $r")
                display("Trying combination: $equation_combinations, error: $(norm(e)), rank: $r")

                if all(abs.(e) .< 1e-10) && r == length(known_solution)
                    display("Found start solution")
                    rotation_intrinsic_system = _rotation_intrinsic_system
                    target_parameters = _target_parameters
                    start_parameters = _start_parameters
                    break
                end
            end

            monodromy_solutions = solutions(
                monodromy_solve(
                    rotation_intrinsic_system,
                    [known_solution],
                    start_parameters;
                    show_progress=false,
                )
            )

            if length(monodromy_solutions) == 0
                continue
            end

            result = solve(
                GeometricHomotopy(
                    rotation_intrinsic_system,
                    start_parameters,
                    target_parameters,
                ),
                monodromy_solutions;
                show_progress=true
            )

            if real_solutions(result) > 32
                break
            end
        end

        display("Performed $(trials) trials")

        @info result

        solution_error, _ = best_overall_solution_by_steps!(
            result,
            problems;
            intrinsic_configuration,
            scene,
        )

        display("Solution error: $solution_error")

        for (i, instance) in enumerate(scene.instances)
            display("View $i")
            print_camera_differences(instance.camera, problems[i].camera)
            display("--------------------")
        end

        print_relative_motion_errors(scene, problems)

        plot_reconstructed_scene(scene, problems; plot_3d = true, raw_3d = false)

        save_2d_figures("assets/test_scenes/hot_pipes/figures/", scene, problems; scene_file_path = "./assets/test_scenes/hot_pipes/scene.json", raw_3d = true)

        display(scene.figure)
    end

    export explore_path, main, monodromy, markers, lights
end