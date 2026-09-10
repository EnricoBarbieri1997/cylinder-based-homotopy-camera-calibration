module EquationSystems
export stack_homotopy_parameters, build_intrinsic_rotation_conic_system, build_intrinsic_rotation_translation_conic_system, build_intrinsic_rotation_translation_conic_system_calibrated, variables_jacobian_rank, joint_jacobian_rank

using ..CylindersBasedCameraResectioning: IMAGE_HEIGHT, IMAGE_WIDTH
using ..Camera: IntrinsicParameters, build_camera_matrix
using ..Space: build_rotation_matrix
using ..Utils: angle_between_3D_directions

using HomotopyContinuation
using LinearAlgebra: det, I, rank, cross

module Problems
using ....Camera: CameraProperties
module IntrinsicParameters
@enum T begin
    focal_length_x = 0b00001
    focal_length_y = 0b00010
    skew = 0b00100
    principal_point_x = 0b01000
    principal_point_y = 0b10000
end
function Base.:|(a::T, b::T)
    UInt8(a) | UInt8(b)
end
function Base.:&(a::T, b::T)
    UInt8(a) & UInt8(b)
end
function Base.:|(a::Any, b::T)
    a | UInt8(b)
end
function Base.:&(a::Any, b::T)
    a & UInt8(b)
end
function Base.:|(a::T, b::Any)
    UInt8(a) | b
end
function Base.:&(a::T, b::Any)
    UInt8(a) & b
end
module has
using ..IntrinsicParameters: focal_length_x, focal_length_y, skew as skewParameter, principal_point_x, principal_point_y
function fₓ(config::Any)
    return (UInt8(config) & focal_length_x) != 0
end
function fᵧ(config::Any)
    return (UInt8(config) & focal_length_y) != 0
end
function skew(config::Any)
    return (UInt8(config) & skewParameter) != 0
end
function cₓ(config::Any)
    return (UInt8(config) & principal_point_x) != 0
end
function cᵧ(config::Any)
    return (UInt8(config) & principal_point_y) != 0
end
end
module Configurations
using ..IntrinsicParameters: focal_length_x, focal_length_y, skew as skew_val, principal_point_x, principal_point_y
@enum T begin
    none = 0
    fₓ = UInt8(focal_length_x)
    fᵧ = UInt8(focal_length_y)
    skew = UInt8(skew_val)
    fₓ_fᵧ = focal_length_x | focal_length_y
    cₓ_cᵧ = principal_point_x | principal_point_y
    fₓ_fᵧ_skew = focal_length_x | focal_length_y | skew_val
    fₓ_fᵧ_skew_cₓ = focal_length_x | focal_length_y | skew_val | principal_point_x
    fₓ_fᵧ_skew_cᵧ = focal_length_x | focal_length_y | skew_val | principal_point_y
    fₓ_fᵧ_cₓ_cᵧ = focal_length_x | focal_length_y | principal_point_x | principal_point_y
    fₓ_fᵧ_skew_cₓ_cᵧ = focal_length_x | focal_length_y | skew_val | principal_point_x | principal_point_y
end
end
end
mutable struct CylinderCameraContoursProblemValidationData
    lines::Array{Float64,2}
    points_at_infinity::Array{Float64,2}
    dualquadrics::Array{Float64,3}
end
mutable struct CylinderCameraContoursProblem
    camera::CameraProperties
    lines::Array{Float64,2}
    noise_free_lines::Array{Float64,2}
    points_at_infinity::Array{Float64,2}
    dualquadrics::Array{Float64, 3}
    validation::CylinderCameraContoursProblemValidationData
    intrinsic_configuration::UInt8
end

end

"""
    inv_upper3(K)

Closed-form inverse of a general upper-triangular 3x3 matrix
K = [a b c; 0 d e; 0 0 f], via back-substitution. Symbolic-safe.
"""
function inv_upper3(K)
    a, b, c = K[1,1], K[1,2], K[1,3]
    d, e    = K[2,2], K[2,3]
    f       = K[3,3]

    return [1/a   -b/(a*d)   (b*e - c*d)/(a*d*f);
            0      1/d       -e/(d*f);
            0      0          1/f]
end

function stack_homotopy_parameters(parameters...)
    stacked_parameters = []
    for parameter in parameters
        stacked_parameters = vcat(stacked_parameters, vec(parameter))
    end
    return stacked_parameters
end

function add_rotation_constraints!(system_to_solve, R)
    push!(system_to_solve, det(R) - 1)
    for eq in vec(R * R' - I)
        push!(system_to_solve, eq)
    end
end

function build_intrinsic_rotation_conic_system(
    problems::Vector{Problems.CylinderCameraContoursProblem};
    equation_combinations = nothing,
)
    problems_count = length(problems)
    if problems_count == 0
        throw(ArgumentError("At least one problem is needed"))
    end

    default_intrinsic = problems[1].camera.intrinsic
    factor = 1.0 / 3000.0
    fₓ = default_intrinsic[1, 1] * factor
    fᵧ = default_intrinsic[2, 2] * factor
    skew = default_intrinsic[1, 2] * factor
    cₓ = default_intrinsic[1, 3] * factor
    cᵧ = default_intrinsic[2, 3] * factor

    system_to_solve = []
    variables::Vector{HomotopyContinuation.ModelKit.Variable} = []
    parameters::Vector{HomotopyContinuation.ModelKit.Variable} = []

    intrinsic_configuration = problems[1].intrinsic_configuration
    if Problems.IntrinsicParameters.has.fₓ(intrinsic_configuration)
        @var fₓ
        push!(variables, fₓ)
    end
    if Problems.IntrinsicParameters.has.fᵧ(intrinsic_configuration)
        @var fᵧ
        push!(variables, fᵧ)
    end
    if Problems.IntrinsicParameters.has.skew(intrinsic_configuration)
        @var skew
        push!(variables, skew)
    end
    if Problems.IntrinsicParameters.has.cₓ(intrinsic_configuration)
        @var cₓ
        push!(variables, cₓ)
    end
    if Problems.IntrinsicParameters.has.cᵧ(intrinsic_configuration)
        @var cᵧ
        push!(variables, cᵧ)
    end

    intrinsic = [
        fₓ skew cₓ;
        0 fᵧ cᵧ;
        0 0 factor
    ]

    Kinv = inv_upper3(intrinsic)
    ω = fₓ^2 * fᵧ^2 * (Kinv' * Kinv)

    ω = expand.(ω)

    display("---w---")
    display(ω)
    display("-------")

    IAC_constraints = []

    for (index_p, problem) in enumerate(problems)
        lines_count = size(problem.lines)[1]
        used_lines = []
        vanishing_points_pointsatinifinity_pairs = []
        # For each line, extract the matching point at inifinity, find another line with the same point at inifinity,
        # if found, calculate the vanishing point and pair it with the point at infinity, and add it to the list of pairs
        for index_l in 1:lines_count
            if (index_l in used_lines)
                continue
            end
            line = problem.lines[index_l, :]
            point_at_infinity = problem.points_at_infinity[index_l, :]
            matching_line_indexes = []
            for index_l2 in index_l:lines_count
                if (index_l2 != index_l) && (problem.points_at_infinity[index_l2, :] == point_at_infinity)
                    push!(matching_line_indexes, index_l2)
                end
            end
            if length(matching_line_indexes) < 1
                continue
            end
            matching_line_index = matching_line_indexes[findfirst(x -> x != index_l, matching_line_indexes)]
            if matching_line_index !== nothing
                matching_line = problem.lines[matching_line_index, :]
                vanishing_point = cross(line, matching_line)
                push!(vanishing_points_pointsatinifinity_pairs, (vanishing_point, point_at_infinity))
            end
            push!(used_lines, index_l)
            for (matching_line_index) in matching_line_indexes
                push!(used_lines, matching_line_index)
            end
        end

        # If at least two pairs of vanishing points and points at infinity are found, add the IAC constraints to the system
        if length(vanishing_points_pointsatinifinity_pairs) >= 2
            for (vp1, pai1) in vanishing_points_pointsatinifinity_pairs
                for (vp2, pai2) in vanishing_points_pointsatinifinity_pairs
                    if vp1 != vp2
                        num = vp1' * ω * vp2
                        den = sqrt((vp1' * ω * vp1) * (vp2' * ω * vp2))
                        θ = angle_between_3D_directions(pai1, pai2)
                        push!(IAC_constraints, num / den - cos(θ))
                    end
                end
            end
        end
    end

    push!(system_to_solve, IAC_constraints...)

    intrinsic_count = count_ones(UInt8(intrinsic_configuration))
    extrinsic_count = size(problems)[1] * 3
    lines_count = intrinsic_count + extrinsic_count - length(IAC_constraints)
    lines_per_problem = zeros(Int64, problems_count)

    problem_index = 1
    for i in 1:lines_count
        lines_per_problem[problem_index] += 1
        problem_index += 1
        if problem_index > problems_count
            problem_index = 1
        end
    end

    if equation_combinations === nothing
        equation_combinations = []
        already_seen = 0
        for (index, problem) in enumerate(problems)
            pick_count = lines_per_problem[index]
            spare_count = size(problem.lines)[1] - pick_count
            
            equation_combinations = vcat(equation_combinations, collect(1:pick_count) .+ already_seen)
            already_seen += (pick_count + spare_count)
        end
    end

    points_at_infinity = vcat([problem.points_at_infinity for problem in problems]...)

    lines = nothing

    for (index, problem) in enumerate(problems)
        lines_to_pick = size(problem.lines)[1]
        _lines = reshape([
                Variable("lines$(index)", i, j)
                for i in 1:lines_to_pick, j in 1:3
            ], lines_to_pick, 3)
        if lines === nothing
            lines = _lines
        else
            lines = vcat(lines, _lines)
        end
    end

    Rs = []

    for (index, problem) in enumerate(problems)
        Rparams = [
            Variable("R$(index)", i) for i in 1:3
        ]
        R = build_rotation_matrix(Rparams..., false)
        push!(Rs, R)

        variables = stack_homotopy_parameters(variables, Rparams)
    end

    for line_index in equation_combinations
        lines_upto_now = 0
        problem_index = 1
        for (index, problem) in enumerate(problems)
            lines_upto_now += size(problem.lines)[1]
            if line_index <= lines_upto_now
                problem_index = index
                break
            end
        end

        line = lines[line_index, :]
        equation = line' * intrinsic * Rs[problem_index] * points_at_infinity[line_index, :]
        push!(system_to_solve, equation)
        parameters = stack_homotopy_parameters(parameters, line)
    end

    a = System(system_to_solve, variables=variables, parameters=parameters)
    return a
end

function build_intrinsic_rotation_translation_conic_system(
    problem::Problems.CylinderCameraContoursProblem;
)
    lines_count = 3
    @var tx ty tz
    @var lines[1:lines_count, 1:3]
    obj = problem.camera
    P = build_camera_matrix(
        obj.intrinsic ./ obj.intrinsic[2, 2],
        obj.rotation_matrix,
        [tx, ty, tz];
        use_rotation_as_is=true
    )

    picked_dualquadrics = zeros(Float64, lines_count, 4, 4)
    picked_dualquadrics_count = 0
    picking_index = 1

    while picked_dualquadrics_count < lines_count
        picked_dualquadrics_count += 1
        picked_dualquadrics[picked_dualquadrics_count, :, :] = problem.dualquadrics[picking_index, :, :]
        picking_index += 2
        if picking_index > size(problem.dualquadrics)[1]
            picking_index = 2
        end
    end

    system_to_solve = []
    parameters::Vector{HomotopyContinuation.ModelKit.Variable} = []
    for i in 1:lines_count
        equation = (lines[i, :]' * P * picked_dualquadrics[i, :, :] * P' * lines[i, :]) / (IMAGE_HEIGHT * IMAGE_WIDTH)
        parameters = stack_homotopy_parameters(parameters, lines[i, :])
        push!(system_to_solve, equation)
    end

    return System(system_to_solve, variables=[tx, ty, tz], parameters=parameters)
end

function build_intrinsic_rotation_translation_conic_system_calibrated(problem::Problems.CylinderCameraContoursProblem)
    new_problem = deepcopy(problem)
    new_problem.camera.intrinsic = Matrix{Float64}(I, 3, 3)
    return build_intrinsic_rotation_translation_conic_system(new_problem)
end

function variables_jacobian(F::System, solution, parameters)
    jacobian = differentiate(F.expressions, F.variables)
    return evaluate(jacobian, F.variables => solution, F.parameters => parameters)
end

function variables_jacobian_rank(F::System, solution, parameters)
    return rank(variables_jacobian(F, solution, parameters))
end

function parameters_jacobian(F::System, solution, parameters)
    jacobian = differentiate(F.expressions, F.parameters)
    return evaluate(jacobian, F.variables => solution, F.parameters => parameters)
end

function parameters_jacobian_rank(F::System, solution, parameters)
    return rank(parameters_jacobian(F, solution, parameters))
end

function joint_jacobian_rank(F::System, solution, parameters)
    return rank(hcat(parameters_jacobian(F, solution, parameters), variables_jacobian(F, solution, parameters)))
end

module Minimization
using ..EquationSystems
using ..Problems

using HomotopyContinuation
using LinearAlgebra: norm

function bestsolution(system, solutions, parameters)
    best_solution = nothing
    best_residual = Inf
    for solution in solutions
        residual = norm(system(solution, parameters))
        if residual < best_residual
            best_residual = residual
            best_solution = solution
        end
    end
    return best_solution
end
function build_intrinsic_rotation_conic_system(
    problems::Vector{Problems.CylinderCameraContoursProblem};
    args...,
)
    sys = EquationSystems.build_intrinsic_rotation_conic_system(problems; args...)
    expressions = sys.expressions
    minimizer = sum(expressions .^ 2)
    variables = sys.variables
    parameters = sys.parameters

    diff = expand.(differentiate(minimizer, variables))

    return System(diff, variables=variables, parameters=parameters)
end

function build_intrinsic_rotation_translation_conic_system(
    problem::Problems.CylinderCameraContoursProblem
)
    sys = EquationSystems.build_intrinsic_rotation_translation_conic_system(problem)
    expressions = sys.expressions
    minimizer = sum(expressions .^ 2)
    variables = sys.variables
    parameters = sys.parameters

    diff = expand.(differentiate(minimizer, variables))

    return System(diff, variables=variables, parameters=parameters)
end
end
end
