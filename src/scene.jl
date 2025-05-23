module Scene

	using ..CylindersBasedCameraResectioning: ASSERTS_ENABLED, IMAGE_HEIGHT, IMAGE_WIDTH
	using ..Geometry: Line, cylinder_rotation_from_axis, homogeneous_line_from_points, homogeneous_to_line, line_to_homogenous, homogeneous_line_intercept, get_cylinder_contours
	using ..Space: RotDeg, transformation, random_transformation, identity_transformation, build_rotation_matrix, position_rotation
	using ..Camera: CameraProperties, IntrinsicParameters, build_intrinsic_matrix, build_camera_matrix, lookat_rotation, random_camera_lookingat_center, is_in_front_of_camera
	using ..Printing: print_camera_differences
	using ..Plotting: initfigure, get_or_add_2d_axis!, clean_plots!, plot_2dpoints, plot_line_2d, plot_image_background, Plot3dCameraInput, plot_3dcamera, plot_3dcamera_rotation, plot_3dcylinders, plot_2dcylinders
	using ..EquationSystems: stack_homotopy_parameters, build_intrinsic_rotation_conic_system, build_intrinsic_rotation_translation_conic_system, build_camera_matrix_conic_system, build_intrinsic_rotation_translation_conic_system_calibrated
	using ..EquationSystems.Problems: CylinderCameraContoursProblem, CylinderCameraContoursProblemValidationData
	using ..EquationSystems.Problems.IntrinsicParameters: Configurations as IntrinsicParametersConfigurations, has as isIntrinsicEnabled
	using ..EquationSystems.Minimization: build_intrinsic_rotation_conic_system as build_intrinsic_rotation_conic_system_minimization
	using ..Utils
	using ..Scene
	using ..Cylinder: CylinderProperties, standard_and_dual as standard_and_dual_cylinder
	using ..Conic: ConicProperties
	using Serialization
	using LinearAlgebra: Diagonal, cross, diagm, deg2rad, dot, I, normalize, pinv, svd
	using Statistics: mean, std
	using HomotopyContinuation, Observables, Polynomials, Rotations
	using Random
	using JSON
	using FileIO

	const MAIN_SET_ERROR_RATIO = 1.0
	const VALIDATION_SET_ERROR_RATIO = 2.0

	struct ParametersSolutionsPair
		start_parameters::Vector{Float64}
		solutions::Union{Vector{Vector{ComplexF64}}, Vector{Vector{Vector{ComplexF64}}}}
	end
	@kwdef mutable struct InstanceConfiguration
		camera::CameraProperties = CameraProperties()
		conics::Vector{ConicProperties} = []
		conics_contours::Array{Float64, 3} = Array{Float64}(undef, 0, 0, 0)
	end
	@kwdef mutable struct SceneData
		cylinders::Vector{CylinderProperties} = []
		instances::Vector{InstanceConfiguration} = []
		figure::Any = nothing
	end

	function add_noise_to_line(line, noise)
		x₁ = 0
		p₁ = [x₁, homogeneous_line_intercept(x₁, line), 1.0] + normalize([rand(Float64, 2) .- 0.5; 0]) * noise
		
		x₂ = 1000
		p₂ = [x₂, homogeneous_line_intercept(x₂, line), 1.0] + normalize([rand(Float64, 2) .- 0.5; 0]) * noise
		
		return homogeneous_line_from_points(p₁, p₂)
	end

	function add_noise_to_lines(line₁, line₂, noise)
		noisy_line₁ = add_noise_to_line(line₁, noise)
		noisy_line₂ = add_noise_to_line(line₂, noise)
		return noisy_line₁, noisy_line₂
	end

	function create_scene_instances_and_problems(;
		random_seed = 7,
		cylinders_random_seed = random_seed,
		number_of_cylinders = 4,
		number_of_instances = 5,
		noise = 0,
		intrinsic_configuration = IntrinsicParametersConfigurations.fₓ_fᵧ_skew_cₓ_cᵧ,
		plot = true,
	)
			Random.seed!(cylinders_random_seed)

			scene = SceneData()
			if plot
				scene.figure = initfigure()
			end

			cylinders = []
			for i in 1:number_of_cylinders
					cylinder = CylinderProperties()
					position = normalize(rand(Float64, 3)) * rand_in_range(0.0, 3.0)
					rotation = rand_in_range((-90, 90), 3)
					cylinder.euler_rotation = rotation

					cylinder.transform = transformation(position, cylinder.euler_rotation)
					radius = rand_in_range((0.2, 1.5), 2)
					cylinder.radiuses = [radius[1], radius[1]] # TODO Support different radiuses for each cylinder

					axis = cylinder.transform * [0; 0; 1; 0]
					axis = axis ./ axis[3]
					axis = axis[1:3]

					standard, dual, singularpoint = standard_and_dual_cylinder(cylinder.transform, cylinder.radiuses)
					cylinder.matrix = standard
					cylinder.dual_matrix = dual
					cylinder.singular_point = singularpoint

					push!(cylinders, cylinder)

					if (ASSERTS_ENABLED)
						# @assert cylinder.matrix * cylinder.dual_matrix ≃ diagm([1, 1, 0, 1]) "(-1) The dual quadric is correct"

						# @assert cylinder.singular_point' * cylinder.matrix * cylinder.singular_point ≃ 0 "(1) Singular point $(1) belongs to the cylinder $(1)"
						# dual_singular_plane = cylinder.transform * reshape([1, 0, 0, -cylinder.radiuses[1]], :, 1)
						# @assert (dual_singular_plane' * cylinder.dual_matrix * dual_singular_plane) ≃ 0 "(2) Perpendicular plane $(1) belongs to the dual cylinder $(1)"

						# @assert (cylinder.matrix * cylinder.singular_point) ≃ [0, 0, 0, 0] "(6) Singular point is right null space of cylinder matrix $(i)"

						# @assert ((dual_singular_plane' * cylinder.singular_point) ≃ 0 && (dual_singular_plane' * cylinder.dual_matrix * dual_singular_plane) ≃ 0) "(7) Singular plane / point and dual quadric constraints $(i)"
						# @assert cylinder.singular_point[4] ≃ 0 "(10) Singular point is at infinity $(i)"
					end
			end

			scene.cylinders = cylinders

			Random.seed!(random_seed)

			instances = []

			focal_length_x = 2250
			focal_length_y = 2666.666667
			skew = 0
			principal_point_x = 540
			principal_point_y = 960

			if (isIntrinsicEnabled.fₓ(intrinsic_configuration))
				focal_length_x = rand_in_range(2500.0, 3000.0)
			end
			if (isIntrinsicEnabled.fᵧ(intrinsic_configuration))
				focal_length_y = rand_in_range(0.8, 1.0) * focal_length_x
			end
			if (isIntrinsicEnabled.skew(intrinsic_configuration))
				skew = rand_in_range(0, 1)
			end
			if (isIntrinsicEnabled.cₓ(intrinsic_configuration))
				principal_point_x = rand_in_range(1280, 1440)
			end
			if (isIntrinsicEnabled.cᵧ(intrinsic_configuration))
				principal_point_y = principal_point_x * (9/16 + rand_in_range(-0.1, 0.1))
			end
			intrinsic = build_intrinsic_matrix(IntrinsicParameters(;
				focal_length_x,
				focal_length_y,
				principal_point_x,
				principal_point_y,
				skew,
			))

			for i in 1:number_of_instances
					instance = InstanceConfiguration()
					position, rotation_matrix = random_camera_lookingat_center()
					quaternion_camera_rotation = QuatRotation(rotation_matrix)
					euler_rotation = rad2deg.(eulerangles_from_rotationmatrix(rotation_matrix))
					camera = CameraProperties(
							position = position,
							euler_rotation = euler_rotation,
							quaternion_rotation = quaternion_camera_rotation,
							intrinsic = intrinsic,
					)

					instance.camera = camera
					
					conics = []
					for i in 1:number_of_cylinders
							camera_matrix = Matrix{Float64}(camera.matrix)
							conic = ConicProperties(
									pinv(camera_matrix') * cylinders[i].matrix * pinv(camera_matrix),
									camera_matrix * cylinders[i].singular_point,
									camera_matrix * cylinders[i].dual_matrix * camera_matrix',
							)
							push!(conics, conic)
					end
					instance.conics = conics

					conics_contours = Array{Float64}(undef, number_of_cylinders, 2, 3)
					for i in 1:number_of_cylinders
							lines = get_cylinder_contours(
									cylinders[i],
									camera
							)
							for (j, line) in enumerate(lines)
									conics_contours[i, j, :] = line

									if (ASSERTS_ENABLED)
										# @assert line' * conics[i].dual_matrix * line ≃ 0 "(3) Line of projected singular plane $(1) belongs to the dual conic $(1)"
										# @assert line' * camera.matrix * cylinders[i].singular_point ≃ 0 "(8) Line $(j) of conic $(i) passes through the projected singular point"
										err = (line' * camera.matrix * cylinders[i].dual_matrix * camera.matrix' * line)
										@assert err ≃ 0 "(9) Line $(j) of conic $(i) is tangent to the projected cylinder. $(err)"
									end
							end
					end
					instance.conics_contours = conics_contours

					push!(instances, instance)
			end

			scene.instances = instances

			intrinsicparamters_count = count_ones(UInt8(intrinsic_configuration))
			problems::Vector{CylinderCameraContoursProblem} = []
			numberoflines_tosolvefor_perinstance = 3 + floor(Int, intrinsicparamters_count/number_of_instances)
			number_of_extra_picks = intrinsicparamters_count % number_of_instances
			for (instance_number, instance) in enumerate(instances)
					conics_contours = instance.conics_contours
					noisy_conic_contours = Array{Float64}(undef, size(conics_contours))
					for i in 1:size(conics_contours)[1]
						line1 = conics_contours[i, 1, :]
						line2 = conics_contours[i, 2, :]
						if noise > 0
							noisy_line_1, noisy_line_2 = add_noise_to_lines(line1, line2, noise)
						else
							noisy_line_1, noisy_line_2 = line1, line2
						end

						noisy_conic_contours[i, 1, :] = noisy_line_1 ./ noisy_line_1[3]
						noisy_conic_contours[i, 2, :] = noisy_line_2 ./ noisy_line_2[3]
					end

					numberoflines_tosolvefor = numberoflines_tosolvefor_perinstance + (instance_number <= number_of_extra_picks ? 1 : 0)

					lines = Matrix{Float64}(undef, numberoflines_tosolvefor, 3)
					noise_free_lines = Matrix{Float64}(undef, numberoflines_tosolvefor, 3)
					points_at_infinity = Matrix{Float64}(undef, numberoflines_tosolvefor, 3)
					dualquadrics = Array{Float64}(undef, numberoflines_tosolvefor, 4, 4)
					line_indexes = Vector{Float64}(undef, numberoflines_tosolvefor)
					possible_picks = collect(1:(number_of_cylinders*2))
					for store_index in (1:numberoflines_tosolvefor)
							line_index = store_index # rand(possible_picks)
							possible_picks = filter(x -> x != line_index, possible_picks)
							i = ceil(Int, line_index / 2)
							j = (line_index - 1) % 2 + 1

							line_indexes[store_index] = line_index

							line = conics_contours[i, j, :]
							noise_free_lines[store_index, :] = normalize(line)
							noisy_line = noisy_conic_contours[i, j, :]
							if noise > 0
								lines[store_index, :] = normalize(noisy_line)
							else
								lines[store_index, :] = normalize(line)
							end
							points_at_infinity[store_index, :] = normalize(cylinders[i].singular_point[1:3])
							dualquadrics[store_index, :, :] = cylinders[i].dual_matrix ./ cylinders[i].dual_matrix[4, 4]
					end
					number_of_spare_lines = length(possible_picks)
					validation_data = CylinderCameraContoursProblemValidationData(
						Matrix{Float64}(undef, number_of_spare_lines, 3),
						Matrix{Float64}(undef, number_of_spare_lines, 3),
						Array{Float64}(undef, number_of_spare_lines, 4, 4),
						Vector{Float64}(undef, numberoflines_tosolvefor)
					)
					for store_index in (1:number_of_spare_lines)
						line_index = possible_picks[1] # rand(possible_picks)
						possible_picks = filter(x -> x != line_index, possible_picks)
						i = ceil(Int, line_index / 2)
						j = (line_index - 1) % 2 + 1

						validation_data.lines[store_index, :] = normalize(noisy_conic_contours[i, j, :])
						validation_data.points_at_infinity[store_index, :] = normalize(cylinders[i].singular_point[1:3])
						validation_data.dualquadrics[store_index, :, :] = cylinders[i].dual_matrix ./ cylinders[i].dual_matrix[4, 4]
						validation_data.line_indexes[store_index] = line_index
					end

					problem_camera = CameraProperties()
					problem_camera.intrinsic = intrinsic
					problem = CylinderCameraContoursProblem(
						problem_camera,
						lines,
						noise_free_lines,
						points_at_infinity,
						dualquadrics,
						line_indexes,
						validation_data,
						UInt8(intrinsic_configuration),
					)
					push!(problems, problem)
			end

			if plot
				plot_scene(scene, problems; noise)
			end

			return scene, problems
	end

	cylinders_names_in_view_file = [
		"red",
		"green",
		"blue"
	]
	function scene_instances_and_problems_from_files(
		scene_file_path,
		views_file_path,
		;plot = true,
		number_of_instances = 2,
	)
			scene_file = nothing
			view_file = nothing

			scene_file = open(scene_file_path, "r") do io
				JSON.parse(io)
			end
			view_file = open(views_file_path, "r") do io
				JSON.parse(io)
			end
			intrinsic_configuration = UInt8(scene_file["configuration"])
			scene = SceneData()
			if plot
				scene.figure = initfigure()
			end

			number_of_cylinders = length(scene_file["cylinders"])

			cylinders = []
			for (i, cylinder_properties) in enumerate(scene_file["cylinders"])
					cylinder = CylinderProperties()
					position = Vector{Float64}(cylinder_properties["position"])
					cylinder.euler_rotation = rad2deg.(cylinder_rotation_from_axis(Vector{Float64}(cylinder_properties["axis"])))

					cylinder.transform = transformation(position, cylinder.euler_rotation)
					radius = Float64(cylinder_properties["radius"])
					cylinder.radiuses = [radius, radius]

					axis = cylinder.transform * [0; 0; 1; 0]
					axis = axis[1:3]

					standard, dual, singularpoint = standard_and_dual_cylinder(cylinder.transform, cylinder.radiuses)
					cylinder.matrix = standard
					cylinder.dual_matrix = dual
					cylinder.singular_point = singularpoint

					push!(cylinders, cylinder)

					# begin #asserts
							# @assert cylinder.matrix * cylinder.dual_matrix ≃ diagm([1, 1, 0, 1]) "(-1) The dual quadric is correct"

							# @assert cylinder.singular_point' * cylinder.matrix * cylinder.singular_point ≃ 0 "(1) Singular point $(1) belongs to the cylinder $(1)"
							# dual_singular_plane = inv(cylinder.transform') * reshape([1, 0, 0, -cylinder.radiuses[1]], :, 1)
							# @assert (dual_singular_plane' * cylinder.dual_matrix * dual_singular_plane) ≃ 0 "(2) Perpendicular plane $(1) belongs to the dual cylinder $(1)"

							# @assert (cylinder.matrix * cylinder.singular_point) ≃ [0, 0, 0, 0] "(6) Singular point is right null space of cylinder matrix $(i)"

							# @assert ((dual_singular_plane' * cylinder.singular_point) ≃ 0 && (dual_singular_plane' * cylinder.dual_matrix * dual_singular_plane) ≃ 0) "(7) Singular plane / point and dual quadric constraints $(i)"
							# @assert cylinder.singular_point[4] ≃ 0 "(10) Singular point is at infinity $(i)"
					# end
			end

			scene.cylinders = cylinders

			instances = []
			intrinsic = Float64.((hcat(scene_file["intrinsics"]...)'))

			number_of_instances = something(number_of_instances, length(instances))

			for instance_index in 1:number_of_instances
					inst = scene_file["cameras"][instance_index]
					instance = InstanceConfiguration()
					projection_rotation_matrix = QuatRotation(inst["R"])
					projection_translation = Vector{Float64}(inst["t"])
					position = -(projection_rotation_matrix') * projection_translation
					quaternion_camera_rotation = projection_rotation_matrix'
					euler_rotation = rad2deg.(eulerangles_from_rotationmatrix(quaternion_camera_rotation))
					camera = CameraProperties(
							position = position,
							euler_rotation = euler_rotation,
							quaternion_rotation = quaternion_camera_rotation,
							intrinsic = intrinsic,
					)

					instance.camera = camera
					instance.conics = []

					conics_contours = Array{Float64}(undef, number_of_cylinders, 2, 3)
					for i in 1:number_of_cylinders
						lines = view_file[instance_index]["lines"][cylinders_names_in_view_file[i]]
						for (j, line) in enumerate(lines)
							p1 = [line[1][1], line[1][2]]
							p2 = [line[2][1], line[2][2]]
							line_homogenous = homogeneous_line_from_points(p1, p2)
							display(line_homogenous ./ line_homogenous[3])
							conics_contours[i, j, :] = line_homogenous

							# if (ASSERTS_ENABLED)
							# 	@assert line_homogenous' * conics[i].dual_matrix * line_homogenous ≃ 0 "(3) Line of projected singular plane $(1) belongs to the dual conic $(1)"
							# 	@assert line_homogenous' * camera.matrix * cylinders[i].singular_point ≃ 0 "(8) Line $(j) of conic $(i) passes through the projected singular point"
							# 	@assert line_homogenous' * camera.matrix * cylinders[i].dual_matrix * camera.matrix' * line_homogenous ≃ 0 "(9) Line $(j) of conic $(i) is tangent to the projected cylinder"
							# end
						end
					end
					instance.conics_contours = conics_contours

					push!(instances, instance)
			end

			scene.instances = instances

			intrinsicparamters_count = count_ones(UInt8(intrinsic_configuration))
			problems::Vector{CylinderCameraContoursProblem} = []
			numberoflines_tosolvefor_perinstance = 3 + floor(Int, intrinsicparamters_count/number_of_instances)
			number_of_extra_picks = intrinsicparamters_count % number_of_instances
			for instance_number in 1:number_of_instances
					instance = instances[instance_number]
					conics_contours = instance.conics_contours

					numberoflines_tosolvefor = numberoflines_tosolvefor_perinstance + (instance_number <= number_of_extra_picks ? 1 : 0)

					lines = Matrix{Float64}(undef, numberoflines_tosolvefor, 3)
					points_at_infinity = Matrix{Float64}(undef, numberoflines_tosolvefor, 3)
					dualquadrics = Array{Float64}(undef, numberoflines_tosolvefor, 4, 4)
					possible_picks = collect(1:(number_of_cylinders*2))
					line_indexes = Vector{Float64}(undef, numberoflines_tosolvefor)
					for store_index in (1:numberoflines_tosolvefor)
						line_index = store_index # rand(possible_picks)
						possible_picks = filter(x -> x != line_index, possible_picks)
						i = ceil(Int, line_index / 2)
						j = (line_index - 1) % 2 + 1

						line_indexes[store_index] = line_index

						line = conics_contours[i, j, :]
						lines[store_index, :] = normalize(line)
						points_at_infinity[store_index, :] = normalize(cylinders[i].singular_point[1:3])
						dualquadrics[store_index, :, :] = cylinders[i].dual_matrix ./ cylinders[i].dual_matrix[4, 4]
					end

					number_of_spare_lines = length(possible_picks)
					validation_data = CylinderCameraContoursProblemValidationData(
						Matrix{Float64}(undef, number_of_spare_lines, 3),
						Matrix{Float64}(undef, number_of_spare_lines, 3),
						Array{Float64}(undef, number_of_spare_lines, 4, 4),
						Vector{Float64}(undef, numberoflines_tosolvefor)
					)
					for store_index in (1:number_of_spare_lines)
						line_index = possible_picks[1]
						possible_picks = filter(x -> x != line_index, possible_picks)
						i = ceil(Int, line_index / 2)
						j = (line_index - 1) % 2 + 1

						validation_data.lines[store_index, :] = normalize(conics_contours[i, j, :])
						validation_data.points_at_infinity[store_index, :] = normalize(cylinders[i].singular_point[1:3])
						validation_data.dualquadrics[store_index, :, :] = cylinders[i].dual_matrix ./ cylinders[i].dual_matrix[4, 4]
						validation_data.line_indexes[store_index] = line_index
					end

					problem_camera = CameraProperties()
					problem_camera_intrinsic = problem_camera.intrinsic
					if (!isIntrinsicEnabled.fₓ(intrinsic_configuration))
						problem_camera_intrinsic[1, 1] = intrinsic[1, 1]
					end
					if (!isIntrinsicEnabled.fᵧ(intrinsic_configuration))
						problem_camera_intrinsic[2, 2] = intrinsic[2, 2]
					end
					if (!isIntrinsicEnabled.skew(intrinsic_configuration))
						problem_camera_intrinsic[1, 2] = intrinsic[1, 2]
					end
					if (!isIntrinsicEnabled.cₓ(intrinsic_configuration))
						problem_camera_intrinsic[1, 3] = intrinsic[1, 3]
					end
					if (!isIntrinsicEnabled.cᵧ(intrinsic_configuration))
						problem_camera_intrinsic[2, 3] = intrinsic[2, 3]
					end
					problem_camera.intrinsic = problem_camera_intrinsic
					problem = CylinderCameraContoursProblem(
							problem_camera,
							lines,
							lines,
							points_at_infinity,
							dualquadrics,
							line_indexes,
							validation_data,
							UInt8(intrinsic_configuration),
					)
					push!(problems, problem)
			end

			if plot
				plot_scene(scene, problems)
				for i in 1:number_of_instances
					camera = scene_file["cameras"][i]
					img = load(joinpath("./", camera["image"]))
					img = rotr90(img)
					plot_image_background(img; axindex=i)
				end
			end

			return scene, problems
	end

	function plot_scene(scene, problems; noise = 0)
		number_of_cylinders = size(scene.cylinders)[1]
		plot_3dcylinders(scene.cylinders)

		for (i, instance) in enumerate(scene.instances)
			camera = instance.camera
			conics = instance.conics
			conics_contours = instance.conics_contours
			plot_3dcamera(camera)
			get_or_add_2d_axis!(i)
			plot_2dpoints([(conic.singular_point) for conic in conics]; axindex = i)
			plot_2dcylinders(conics_contours, alpha=0.5; axindex = i)
			centers = [camera.matrix * [position_rotation(cylinder.transform)[1]; 1] for cylinder in scene.cylinders]
			top_bound = [camera.matrix * [(position_rotation(cylinder.transform)[1] + [0.0, 0.0, -cylinder.radiuses[1]]); 1] for cylinder in scene.cylinders]
			bottom_bound = [camera.matrix * [(position_rotation(cylinder.transform)[1] + [0.0, 0.0, cylinder.radiuses[1]]); 1] for cylinder in scene.cylinders]
			plot_2dpoints(centers; axindex = i)
			plot_2dpoints(top_bound; axindex = i)
			plot_2dpoints(bottom_bound; axindex = i)
		end
		if (noise > 0)
			for (i, problem) in enumerate(problems)
				ordered_contours = zeros(size(scene.cylinders)[1] * 2, 3)
				is_first_line = fill(true, size(scene.cylinders)[1])
				for j in 1:size(problem.lines)[1]
					cylinder_index = findfirst((cylinder) -> normalize(cylinder.singular_point[1:3]) == problem.points_at_infinity[j, :], scene.cylinders)
					new_line_index = cylinder_index * 2 - (is_first_line[cylinder_index] ? 1 : 0)
					is_first_line[cylinder_index] = false
					ordered_contours[new_line_index, :, :] = problem.lines[j, :, :]
				end
				noisy_contours = vcat(ordered_contours)
				noisy_contours = reshape(noisy_contours, 2, number_of_cylinders, 3)
				noisy_contours = permutedims(noisy_contours, (2,1,3))
				plot_2dcylinders(noisy_contours; linestyle=:dashdotdot, axindex = i)
			end
		end
	end

	function plot_interactive_scene(;
			scene,
			problems,
			noise = 0,
			observable_instances,
			figure,
		)
		on(observable_instances) do instances
			try
				observable_scene = SceneData(;
					figure,
					cylinders = scene.cylinders,
					instances,
				)
				clean_plots!()
				plot_scene(observable_scene, problems; noise)
				for (i, instance) in enumerate(instances)
					plot_3dcamera_rotation(camera; axindex = i)
					plot_3dcamera_rotation(camera; color=:green, axindex = i)
				end
			catch e
				@error e
			end
		end
	end

	function split_intrinsic_rotation_parameters(
			solution,
			intrinsic_configuration = IntrinsicParametersConfigurations.fₓ_fᵧ_skew_cₓ_cᵧ;
			starting_camera = CameraProperties()
	)
			intrinsicparamters_count = count_ones(UInt8(intrinsic_configuration))

			intrinsics_solution = solution[1:intrinsicparamters_count]
			focal_length_x = starting_camera.intrinsic[1, 1]
			focal_length_y = starting_camera.intrinsic[2, 2]
			skew = starting_camera.intrinsic[1, 2]
			principal_point_x = starting_camera.intrinsic[1, 3]
			principal_point_y = starting_camera.intrinsic[2, 3]
			factor = 1

			intrinsic_solution_index = 1
			if (isIntrinsicEnabled.fₓ(intrinsic_configuration))
					focal_length_x = intrinsics_solution[intrinsic_solution_index]
					intrinsic_solution_index += 1
			end
			if (isIntrinsicEnabled.fᵧ(intrinsic_configuration))
					focal_length_y = 1
					factor = intrinsics_solution[intrinsic_solution_index]
					if (!isIntrinsicEnabled.fₓ(intrinsic_configuration))
						focal_length_x = 1
					end
					intrinsic_solution_index += 1
			end
			if (isIntrinsicEnabled.skew(intrinsic_configuration))
					skew = intrinsics_solution[intrinsic_solution_index]
					intrinsic_solution_index += 1
			end
			if (isIntrinsicEnabled.cₓ(intrinsic_configuration))
					principal_point_x = intrinsics_solution[intrinsic_solution_index]
					intrinsic_solution_index += 1
			end
			if (isIntrinsicEnabled.cᵧ(intrinsic_configuration))
					principal_point_y = intrinsics_solution[intrinsic_solution_index]
					intrinsic_solution_index += 1
			end

			if (isIntrinsicEnabled.fₓ(intrinsic_configuration) || isIntrinsicEnabled.fᵧ(intrinsic_configuration))
				focal_length_x = focal_length_x / factor
			end
			if (isIntrinsicEnabled.fᵧ(intrinsic_configuration))
				focal_length_y = focal_length_y / factor
			end
			if (isIntrinsicEnabled.skew(intrinsic_configuration))
				skew = skew / factor
			end
			if (isIntrinsicEnabled.cₓ(intrinsic_configuration))
				principal_point_x = principal_point_x / factor
			end
			if (isIntrinsicEnabled.cᵧ(intrinsic_configuration))
				principal_point_y = principal_point_y / factor
			end

			# Spurious solutions
			if (factor ≃ 0 || focal_length_x ≃ 0 || focal_length_y ≃ 0)
				throw(ArgumentError("Spurious solution"))
			end

			intrinsic = build_intrinsic_matrix(IntrinsicParameters(
					focal_length_x = focal_length_x,
					focal_length_y = focal_length_y,
					principal_point_x = principal_point_x,
					principal_point_y = principal_point_y,
					skew = skew,
			))
			intrinsic_correction = Matrix{Float64}(I, 3, 3)
			if (focal_length_x < 0)
					intrinsic_correction *= [
							-1 0 0;
							0 1 0;
							0 0 1;
					]
			end
			if (focal_length_y < 0 && skew <= 0)
					intrinsic_correction *= [
							1 0 0;
							1 -1 0;
							0 0 1;
					]
			end
			intrinsic_solution = intrinsic * intrinsic_correction
			rotations_solution = solution[(intrinsicparamters_count + 1):end]

			return intrinsic_solution, rotations_solution, intrinsic_correction
	end

	function camera_from_solution(
		intrinsic,
		rotations_solution,
		intrinsic_correction,
		index,
	)
		quat = [1; rotations_solution[(index-1)*3+1:index*3]]
		quat = quat / norm(quat)
		camera_extrinsic_rotation = QuatRotation(quat) * inv(intrinsic_correction)

		return CameraProperties(
			euler_rotation = rad2deg.(eulerangles_from_rotationmatrix(camera_extrinsic_rotation')),
			quaternion_rotation = camera_extrinsic_rotation',
			intrinsic = intrinsic,
		)
	end

	function intrinsic_rotation_system_setup(
		problems
	)
			rotation_intrinsic_system = build_intrinsic_rotation_conic_system(
				problems
			)
			parameters = []
			for problem in problems
				parameters = stack_homotopy_parameters(
					parameters,
					problem.lines,
				)
			end
			parameters = convert(Vector{Float64}, parameters)

			return rotation_intrinsic_system, parameters
	end

	function intrinsic_rotation_problem_error(problem, intrinsic, camera_extrinsic_rotation)
		current_error = 0.0
		for (line, point_at_infinity) in zip(
			eachslice(problem.lines, dims=1),
			eachslice(problem.points_at_infinity, dims=1)
		)
			eq = line' * (intrinsic ./ intrinsic[2, 2]) * camera_extrinsic_rotation * point_at_infinity
			current_error += abs(eq)
		end
		return current_error
	end

	function best_intrinsic_rotation_system_solution!(
			result,
			problems;
			start_error = Inf,
			intrinsic_configuration = IntrinsicParametersConfigurations.fₓ_fᵧ_skew_cₓ_cᵧ,
			scene = nothing,
			excluded_solutions = [],
	)
			solution_error = start_error
			solutions_to_try = [real(sol) for sol in solutions(result) if !any(isnan, real(sol))]
			all_possible_solutions = []
			best_solution = nothing
			starting_camera = problems[1].camera
			for solution in solutions_to_try
					if (any(x -> x ≃ solution, excluded_solutions))
						continue
					end
					intrinsic = rotations_solution = intrinsic_correction = nothing
					try
						intrinsic, rotations_solution, intrinsic_correction = split_intrinsic_rotation_parameters(
							solution,
							intrinsic_configuration;
							starting_camera
						)
					catch
						continue
					end

					current_error = 0
					if (!isnothing(scene))
							current_error += sum(intrinsic_difference(
								intrinsic,
								scene.instances[1].camera.intrinsic,
							))
							# display("Intrinsic difference: $(intrinsic_difference(
							# 	intrinsic,
							# 	scene.instances[1].camera.intrinsic,
							# ))")
					end
					possible_cameras = []
					for i in eachindex(problems)
							quat = [1; rotations_solution[(i-1)*3+1:i*3]]
							quat = quat / norm(quat)
							camera_extrinsic_rotation = (QuatRotation(quat) * inv(intrinsic_correction))'

							possible_camera = camera_from_solution(
								intrinsic,
								rotations_solution,
								intrinsic_correction,
								i,
							)
							push!(possible_cameras, possible_camera)
							problem = problems[i]

							if (isnothing(scene))
								current_error += MAIN_SET_ERROR_RATIO * intrinsic_rotation_problem_error(problem, intrinsic, camera_extrinsic_rotation)
								current_error += VALIDATION_SET_ERROR_RATIO * intrinsic_rotation_problem_error(problem.validation, intrinsic, camera_extrinsic_rotation)
							else
								current_error += rotations_difference(
									possible_camera.quaternion_rotation,
									scene.instances[i].camera.quaternion_rotation,
								)
							end
					end
					push!(all_possible_solutions, possible_cameras[1])

					if (current_error < solution_error)
							# display("New best solution error: $(current_error)")
							# display("New best solution: $(solution)")
							solution_error = current_error
							best_solution = solution
							for (i, problem) in enumerate(problems)
									problem.camera = possible_cameras[i]
							end
					end
			end

			# display("Best solution error: $(solution_error)")
			# display("Best solution: $(solutions_to_try)")

			if (!isnothing(best_solution))
				paths = path_results(result)
				best_path = paths[findall(x -> real.(x.solution) == best_solution, paths)[1]]
				# display("The best starting solution was $(best_path.start_solution)")
			end

			return solution_error, all_possible_solutions, best_solution
	end

	function intrinsic_rotation_translation_system_setup(problem; calibrate = false)
			if (calibrate)
				translation_system = build_intrinsic_rotation_translation_conic_system_calibrated(
					problem
				)
				lines = hcat([(problem.camera.intrinsic' * line) for line in eachslice(problem.lines, dims=1)]...)'
				if (ASSERTS_ENABLED)
					errs = []
					for (i, line) in enumerate(eachslice(problem.lines, dims=1))
						eq = line' * problem.camera.matrix[1:3, 1:3] * problem.points_at_infinity[i, :]
						push!(errs, eq)
					end
					for (i, line) in enumerate(eachslice(lines, dims=1))
						eq = line' * problem.camera.quaternion_rotation' * problem.points_at_infinity[i, :]
						@assert eq ≃ errs[i] "Camera calibration not successful for line $(i) with error $(eq)"
					end
				end
				parameters = stack_homotopy_parameters(lines[1:3, 1:3])
			else
				translation_system = build_intrinsic_rotation_translation_conic_system(
					problem
				)
				parameters = stack_homotopy_parameters(problem.lines[1:3, 1:3])
			end

			return translation_system, parameters
	end

	function problem_reprojection_error(scene, problem; camera = nothing)
		cylinders = scene.cylinders
		number_of_cylinders = length(cylinders)
		error = 0
		for i in 1:number_of_cylinders
			lines = get_cylinder_contours(
				cylinders[i],
				camera
			)
			for (j, line) in enumerate(lines)
				line_index = (i-1)*2+j
				line_position = findfirst(==(line_index), problem.line_indexes)
				if (!isnothing(line_position))
					line_truth = normalize(problem.lines[line_position,:])
					line_calculated = normalize(line)
					error += (norm(line_calculated - line_truth))^2
				end
			end
		end
		return error
	end

	function z_axis_penalty(camera)
		rotation = camera.euler_rotation
		roll_rad = deg2rad(rotation[3])
		return abs(π - abs(mod(roll_rad + π, 2π) - π))
	end

	function intrinsic_rotation_translation_problem_error(problem, intrinsic, camera_matrix)
		current_error = 0
		for (line, dualquadric) in zip(
			eachslice(problem.lines, dims=1),
			eachslice(problem.dualquadrics, dims=1)
		)
			calibrated_line = intrinsic' * line
			eq = calibrated_line' * camera_matrix * dualquadric * camera_matrix' * calibrated_line
			current_error += abs(eq)
		end
		return current_error
	end

	function best_intrinsic_rotation_translation_system_solution!(
			result,
			problem;
			scene = nothing,
			reference_instance = nothing,
			use_plain_errors = false,
	)
			solution_error = Inf
			solutions_to_try = real_solutions(result)
			intrinsic = problem.camera.intrinsic ./ problem.camera.intrinsic[2, 2]
			
			val_errors = Float64[]
			in_front_count = 0
			low_val_error_count = 0

			for solution in solutions_to_try
				tx, ty, tz = solution
				test_problem = deepcopy(problem)
				test_problem.camera.position = [tx, ty, tz]
				# print("{\n\"position\": [$(tx), $(ty), $(tz)],")

				if (!isnothing(scene))
					is_in_front = is_in_front_of_camera(test_problem.camera)
					if is_in_front
						in_front_count += 1
					end
					if (!is_in_front)
						continue
					end
					training_error = MAIN_SET_ERROR_RATIO * problem_reprojection_error(
						scene,
						test_problem;
						camera = test_problem.camera
					)
					validation_error = VALIDATION_SET_ERROR_RATIO * problem_reprojection_error(
						scene,
						test_problem.validation;
						camera = test_problem.camera
					)
					# print("\"validation_error:\": \"$(b)\"\n},")
					push!(val_errors, validation_error)
					if validation_error < 1e-6
						low_val_error_count += 1
					end
					current_error = training_error + validation_error
				elseif (!isnothing(reference_instance))
					current_error = translations_difference(
						test_problem.camera.position,
						reference_instance.camera.position
					)
				else
					camera_matrix = build_camera_matrix(
							Matrix{Float64}(I, 3, 3),
							test_problem.camera.quaternion_rotation,
							test_problem.camera.position
					)

					current_error = MAIN_SET_ERROR_RATIO * intrinsic_rotation_translation_problem_error(
						test_problem,
						intrinsic,
						camera_matrix
					)
					current_error += VALIDATION_SET_ERROR_RATIO * intrinsic_rotation_translation_problem_error(
						test_problem.validation,
						intrinsic,
						camera_matrix
					)
				end

				# display("Current error: $(current_error)")

				if (current_error < solution_error)
					solution_error = current_error
					problem.camera.position = test_problem.camera.position
				end
			end

			if (!use_plain_errors)
				display("correct")
				mean_val = mean(val_errors)
				std_val = std(val_errors)

				solution_error = -(2.0 * low_val_error_count +
				1.5 * in_front_count +
				-1.0 * std_val +
				-1.0 * mean_val)
			end

			return solution_error
	end

	function best_overall_solution!(
			result,
			scene,
			problems;
			start_error = Inf,
			intrinsic_configuration = IntrinsicParametersConfigurations.fₓ_fᵧ_skew_cₓ_cᵧ
	)
			solution_error = start_error
			solutions_to_try = real_solutions(result)
			all_possible_solutions = []
			starting_camera = problems[1].camera
			for solution in solutions_to_try
				intrinsic = rotations_solution = intrinsic_correction = nothing
				try
					intrinsic, rotations_solution, intrinsic_correction = split_intrinsic_rotation_parameters(
						solution,
						intrinsic_configuration;
						starting_camera
					)
				catch e
					@error e
					continue
				end

				possible_cameras = []
				current_error = 0

				for (i, problem) in enumerate(problems)
						individual_problem_error = 0
						quat = [1; rotations_solution[(i-1)*3+1:i*3]]
						quat = quat / norm(quat)
						camera_extrinsic_rotation = (QuatRotation(quat) * inv(intrinsic_correction))'
						euler_rotation = eulerangles_from_rotationmatrix(camera_extrinsic_rotation)
						# print("{\n\"rotation\": [$(euler_rotation[1]), $(euler_rotation[2]), $(euler_rotation[3])],")
						# print("\"true_error\": \"$(rotations_difference(scene.instances[i].camera.quaternion_rotation, camera_extrinsic_rotation))\",")
						# print("\"options\": [\n")

						problem_upto_translation = CylinderCameraContoursProblem(
								CameraProperties(
										euler_rotation = rad2deg.(eulerangles_from_rotationmatrix(camera_extrinsic_rotation)),
										quaternion_rotation = camera_extrinsic_rotation,
										intrinsic = intrinsic,
								),
								problem.lines,
								problem.noise_free_lines,
								problem.points_at_infinity,
								problem.dualquadrics,
								problem.line_indexes,
								problem.validation,
								problem.intrinsic_configuration,
						)

						current_error += 2.0 * z_axis_penalty(problem_upto_translation.camera)

						translation_system, parameters = intrinsic_rotation_translation_system_setup(problem_upto_translation)
						solver, starts = solver_startsolutions(
							translation_system;
							target_parameters = parameters,
							start_system = :total_degree
						)
						# display("starts: $(starts)")

						try
							translation_result = solve(
								translation_system;
								target_parameters = parameters,
								start_system = :total_degree,
								# show_progress = false
							)
							# @info result

							current_error += best_intrinsic_rotation_translation_system_solution!(
									translation_result,
									problem_upto_translation;
									scene
							)

							possible_camera = problem_upto_translation.camera
							push!(possible_cameras, possible_camera)
						catch e
							Base.showerror(stdout, e)
							Base.show_backtrace(stdout, catch_backtrace())
							current_error = Inf
						end
				end
				# print("]\n},\n")
				if current_error == Inf
						continue
				end
				push!(all_possible_solutions, Dict(
						"camera" => possible_cameras[1],
						"solution" => current_error,
				))
					

				if (current_error < solution_error)
					solution_error = current_error
					for (i, problem) in enumerate(problems)
						problem.camera = possible_cameras[i]
					end
				end
			end

			return solution_error, all_possible_solutions
	end

	function averaged_solution!(
		result,
		scene,
		problems;
		previous_solution = nothing,
		intrinsic_configuration = IntrinsicParametersConfigurations.fₓ_fᵧ_skew_cₓ_cᵧ
	)
		solutions_to_try = real_solutions(result)
		starting_camera = problems[1].camera
		
		solutions_splitted = []
		for solution in solutions_to_try
			intrinsic = rotations_solution = intrinsic_correction = nothing
			try
				intrinsic, rotations_solution, intrinsic_correction = split_intrinsic_rotation_parameters(
					solution,
					intrinsic_configuration;
					starting_camera
				)
			catch e
				@error e
				continue
			end
			push!(solutions_splitted, Dict(
				"intrinsic" => intrinsic,
				"rotations" => rotations_solution,
				"correction" => intrinsic_correction,
			))
		end

		all_possible_solutions = []
		for (i, problem) in enumerate(problems)
			for solution in solutions_splitted
				solution_error = 0.0

				intrinsic = solution["intrinsic"]
				rotations_solution = solution["rotations"]
				intrinsic_correction = solution["correction"]

				quat = [1; rotations_solution[(i-1)*3+1:i*3]]
				quat = quat / norm(quat)
				camera_extrinsic_rotation = (QuatRotation(quat) * inv(intrinsic_correction))'
				euler_rotation = eulerangles_from_rotationmatrix(camera_extrinsic_rotation)
				# print("{\n\"rotation\": [$(euler_rotation[1]), $(euler_rotation[2]), $(euler_rotation[3])],")
				# print("\"true_error\": \"$(rotations_difference(scene.instances[i].camera.quaternion_rotation, camera_extrinsic_rotation))\",")
				# print("\"options\": [\n")

				problem_upto_translation = CylinderCameraContoursProblem(
					CameraProperties(
							euler_rotation = rad2deg.(eulerangles_from_rotationmatrix(camera_extrinsic_rotation)),
							quaternion_rotation = camera_extrinsic_rotation,
							intrinsic = intrinsic,
					),
					problem.lines,
					problem.noise_free_lines,
					problem.points_at_infinity,
					problem.dualquadrics,
					problem.line_indexes,
					problem.validation,
					problem.intrinsic_configuration,
				)

				solution_error += 2.0 * z_axis_penalty(problem_upto_translation.camera)

				translation_system, parameters = intrinsic_rotation_translation_system_setup(problem_upto_translation)
				solver, starts = solver_startsolutions(
					translation_system;
					target_parameters = parameters,
					start_system = :total_degree
				)
				# display("starts: $(starts)")

				try
					translation_result = solve(
						translation_system;
						target_parameters = parameters,
						start_system = :total_degree,
						# show_progress = false
					)
					# @info result

					solution_error += best_intrinsic_rotation_translation_system_solution!(
							translation_result,
							problem_upto_translation;
							scene
					)
				catch e
					Base.showerror(stdout, e)
					Base.show_backtrace(stdout, catch_backtrace())
					solution_error = Inf
				end
				# print("]\n},\n")
				if solution_error == Inf
						continue
				end

				push!(all_possible_solutions, Dict(
					"camera" => problem_upto_translation.camera,
					"error" => solution_error,
				))
			end

			min_error = minimum(sol["error"] for sol in all_possible_solutions)
			max_error = maximum(sol["error"] for sol in all_possible_solutions)
			for sol in all_possible_solutions
				sol["error"] = sol["error"] - min_error
				if (sol["error"] < 0)
					sol["error"] = 0.0
				end
				sol["error"] = sol["error"] / (max_error - min_error)
				sol["score"] = 1 - sol["error"]
			end

			score_sum = sum(sol["score"] for sol in all_possible_solutions)
			for sol in all_possible_solutions
				sol["ratio"] = sol["score"] / score_sum
			end

			camera = CameraProperties()
			camera.intrinsic = sum((sol["ratio"] * sol["camera"].intrinsic) for sol in all_possible_solutions)
			camera.quaternion_rotation = sum((sol["ratio"] * sol["camera"].quaternion_rotation) for sol in all_possible_solutions)
			camera.position = sum((sol["ratio"] * sol["camera"].position) for sol in all_possible_solutions)

			if (!isnothing(previous_solution))
				previous_problem_solution_camera = previous_solution[i].camera
				camera.quaternion_rotation = (previous_problem_solution_camera.quaternion_rotation + camera.quaternion_rotation) / 2
				camera.position = (previous_problem_solution_camera.position + camera.position) / 2
			end

			camera.euler_rotation = rad2deg.(eulerangles_from_rotationmatrix(camera.quaternion_rotation))
			problem.camera = camera
		end

		intrinsic = sum(problem.camera.intrinsic for problem in problems) ./ length(problems)
		if (!isnothing(previous_solution))
			previous_problem_solution_camera = previous_solution[1].camera
			intrinsic = (previous_problem_solution_camera.intrinsic + intrinsic) / 2
		end
		for problem in problems
			problem.camera.intrinsic = intrinsic
		end
		return deepcopy(problems), all_possible_solutions
	end

	function best_overall_solution_by_steps!(
		result,
		problems;
		start_error = Inf,
		intrinsic_configuration = IntrinsicParametersConfigurations.fₓ_fᵧ_skew_cₓ_cᵧ,
		scene = nothing,
		validation_cylinders = nothing,
	)
		valid_solution_found = false
		excluded_solutions = []
		solution_error, all_possible_solutions = nothing, nothing
		tryied_solutons = 0
		problems_to_solve = nothing
		while (!valid_solution_found && tryied_solutons < length(real_solutions(result)))
			valid_solution_found = true
			tryied_solutons += 1
			display("Solution: $(tryied_solutons)")
			# display(excluded_solutions)
			problems_to_solve = deepcopy(problems)
			solution_error, all_possible_solutions, best_solution = best_intrinsic_rotation_system_solution!(
				result,
				problems_to_solve;
				start_error=start_error,
				intrinsic_configuration,
				scene,
				excluded_solutions,
			)

			for (i, problem) in enumerate(problems_to_solve)
				display("Problem $i")
				translation_system, parameters = intrinsic_rotation_translation_system_setup(
					problem;
					calibrate = true
				)
				try
					translation_result = solve(
							translation_system;
							target_parameters = parameters,
							start_system = :total_degree,
					)
					@info translation_result

					best_intrinsic_rotation_translation_system_solution!(
						translation_result,
						problem;
						scene,
						use_plain_errors = true,
					)

					if (!isnothing(validation_cylinders))
						for cylinder in validation_cylinders
							get_cylinder_contours(
								cylinder,
								problem.camera
							)
						end
					end
				catch e
					Base.showerror(stdout, e)
					Base.show_backtrace(stdout, catch_backtrace())

					# push!(excluded_solutions, best_solution)
					# valid_solution_found = false
				end
			end
		end

		if (valid_solution_found)
			for (i, problem) in enumerate(problems)
				problem.camera = problems_to_solve[i].camera
			end
		end

		return solution_error, all_possible_solutions
	end

	function plot_reconstructed_scene(scene, problems)
			number_of_cylinders = size(scene.cylinders)[1]
			for (i, problem) in enumerate(problems)
					plot_3dcamera(problem.camera, :green)
					reconstructued_contours = Array{Float64}(undef, number_of_cylinders, 2, 3)
					for i in 1:number_of_cylinders
							lines = get_cylinder_contours(
									scene.cylinders[i],
									problem.camera
							)
							for (j, line) in enumerate(lines)
									reconstructued_contours[i, j, :] = line
							end
					end

					plot_2dcylinders(reconstructued_contours, linestyle=:dash; axindex = i)
			end
	end
end