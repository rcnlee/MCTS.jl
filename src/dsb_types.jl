"""
MCTS solver with DSB

Fields:

    depth::Int64:
        Maximum rollout horizon and tree depth.
        default: 10

    exploration_constant::Float64:
        Specified how much the solver should explore.
        In the UCB equation, Q + c*sqrt(log(t/N)), c is the exploration constant.
        default: 1.0

    n_iterations::Int64
        Number of iterations during each action() call.
        default: 100

    max_time::Float64
        Maximum amount of CPU time spent iterating through simulations.
        default: Inf

    k_action::Float64
    alpha_action::Float64
    k_state::Float64
    alpha_state::Float64
        These constants control the double progressive widening. A new state
        or action will be added if the number of children is less than or equal to kN^alpha.
        defaults: k:10, alpha:0.5

    keep_tree::Bool
        If true, store the tree in the planner for reuse at the next timestep (and every time it is used in the future). There is a computational cost for maintaining the state dictionary necessary for this.
        default: false

    enable_action_pw::Bool
        If true, enable progressive widening on the action space; if false just use the whole action space.
        default: true

    check_repeat_state::Bool
    check_repeat_action::Bool
        When constructing the tree, check whether a state or action has been seen before (there is a computational cost to maintaining the dictionaries necessary for this)
        default: true

    rng::AbstractRNG:
        Random number generator

    estimate_value::Any (rollout policy)
        Function, object, or number used to estimate the value at the leaf nodes.
        If this is a function `f`, `f(mdp, s, depth)` will be called to estimate the value.
        If this is an object `o`, `estimate_value(o, mdp, s, depth)` will be called.
        If this is a number, the value will be set to that number.
        default: RolloutEstimator(RandomSolver(rng))

    init_Q::Any
        Function, object, or number used to set the initial Q(s,a) value at a new node.
        If this is a function `f`, `f(mdp, s, a)` will be called to set the value.
        If this is an object `o`, `init_Q(o, mdp, s, a)` will be called.
        If this is a number, Q will always be set to that number.
        default: 0.0

    init_N::Any
        Function, object, or number used to set the initial N(s,a) value at a new node.
        If this is a function `f`, `f(mdp, s, a)` will be called to set the value.
        If this is an object `o`, `init_N(o, mdp, s, a)` will be called.
        If this is a number, N will always be set to that number.
        default: 0

    next_action::Any
        Function or object used to choose the next action to be considered for progressive widening.
        The next action is determined based on the MDP, the state, `s`, and the current `DSBStateNode`, `snode`.
        If this is a function `f`, `f(mdp, s, snode)` will be called to set the value.
        If this is an object `o`, `next_action(o, mdp, s, snode)` will be called.
        default: RandomActionGenerator(rng)
"""
mutable struct DSBSolver <: AbstractMCTSSolver
    depth::Int
    exploration_constant::Float64
    n_iterations::Int
    max_time::Float64
    k_action::Float64
    alpha_action::Float64
    k_state::Float64
    alpha_state::Float64

    #dsb
    r0_action::Float64
    lambda_action::Float64
    r0_state::Float64
    lambda_state::Float64

    keep_tree::Bool
    enable_action_pw::Bool
    check_repeat_state::Bool
    check_repeat_action::Bool
    rng::AbstractRNG
    estimate_value::Any
    init_Q::Any
    init_N::Any
    next_action::Any
    listener::Any
end

"""
    DSBSolver()

Use keyword arguments to specify values for the fields
"""
function DSBSolver(;depth::Int=10,
                    exploration_constant::Float64=1.0,
                    n_iterations::Int=100,
                    max_time::Float64=Inf,
                    k_action::Float64=10.0,
                    alpha_action::Float64=0.5,
                    k_state::Float64=10.0,
                    alpha_state::Float64=0.5,

                    r0_action::Float64=5.0,
                    lambda_action::Float64=0.5,
                    r0_state::Float64=5.0,
                    lambda_state::Float64=0.5,

                    keep_tree::Bool=false,
                    enable_action_pw::Bool=true,
                    check_repeat_state::Bool=true,
                    check_repeat_action::Bool=true,
                    rng::AbstractRNG=Base.GLOBAL_RNG,
                    estimate_value::Any = RolloutEstimator(RandomSolver(rng)),
                    init_Q::Any = 0.0,
                    init_N::Any = 0,
                    next_action::Any = RandomActionGenerator(rng),
                    listener::Any=nothing)
    DSBSolver(depth, exploration_constant, n_iterations, max_time, k_action, alpha_action, k_state, alpha_state, r0_action, lambda_action, r0_state, lambda_state, keep_tree, enable_action_pw, check_repeat_state, check_repeat_action, rng, estimate_value, init_Q, init_N, next_action, listener)
end

mutable struct DSBPlanner{P<:Union{MDP,POMDP}, S, A, SE, NA, RNG} <: AbstractMCTSPlanner{P}
    solver::DSBSolver
    mdp::P
    tree::Nullable{DPWTree{S,A}}
    solved_estimate::SE
    next_action::NA
    rng::RNG
end

function DSBPlanner{S,A}(solver::DSBSolver, mdp::Union{POMDP{S,A},MDP{S,A}})
    se = convert_estimator(solver.estimate_value, solver, mdp)
    return DSBPlanner(solver,
                      mdp,
                      Nullable{DPWTree{S,A}},
                      se,
                      solver.next_action,
                      solver.rng
                     )
end

Base.srand(p::DSBPlanner, seed) = srand(p.rng, seed)
