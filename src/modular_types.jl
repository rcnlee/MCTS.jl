include("modules/ModularBandit.jl")

"""
Modular MCTS solver 

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

    tree_in_info::Bool:
        If true, return the tree in the info dict when action_info is called. False by default because it can use a lot of memory if histories are being saved.
        default: false

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
        The next action is determined based on the MDP, the state, `s`, and the current `ModularStateNode`, `snode`.
        If this is a function `f`, `f(mdp, s, snode)` will be called to set the value.
        If this is an object `o`, `next_action(o, mdp, s, snode)` will be called.
        default: RandomActionGenerator(rng)

    default_action::Any
        Function, action, or Policy used to determine the action if POMCP fails with exception `ex`.
        If this is a Function `f`, `f(pomdp, belief, ex)` will be called.
        If this is a Policy `p`, `action(p, belief)` will be called.
        If it is an object `a`, `default_action(a, pomdp, belief, ex)` will be called, and if this method is not implemented, `a` will be returned directly.
        default: `ExceptionRethrow()`
"""
mutable struct ModularSolver <: AbstractMCTSSolver
    depth::Int
    n_iterations::Int
    max_time::Float64
    bandit::ModularBandit
    keep_tree::Bool
    check_repeat_state::Bool
    tree_in_info::Bool
    rng::AbstractRNG
    estimate_value::Any
    init_Q::Any
    init_N::Any
    next_action::Any
    default_action::Any
    track_best_path::Bool
end

"""
    ModularSolver()

Use keyword arguments to specify values for the fields
"""
function ModularSolver(;depth::Int=10,
                    n_iterations::Int=100,
                    max_time::Float64=Inf,
                    bandit::ModularBandit=DPWBandit(),
                    keep_tree::Bool=false,
                    check_repeat_state::Bool=true,
                    tree_in_info::Bool=false,
                    rng::AbstractRNG=Base.GLOBAL_RNG,
                    estimate_value::Any = RolloutEstimator(RandomSolver(rng)),
                    init_Q::Any=0.0,
                    init_N::Any=0,
                    next_action::Any=RandomActionGenerator(rng),
                    default_action::Any=ExceptionRethrow(),
                    track_best_path::Bool=false
                   )
    ModularSolver(depth, n_iterations, max_time, bandit, keep_tree, check_repeat_state, tree_in_info, rng, estimate_value, init_Q, init_N, next_action, default_action, track_best_path)
end

mutable struct ModularTree{S,A}
    # for each state node
    total_n::Vector{Int}
    children::Vector{Vector{Int}}
    s_labels::Vector{S}
    s_lookup::Dict{S, Int}

    # for each state-action node
    n::Vector{Int}
    q::Vector{Float64}
    transitions::Vector{Vector{Tuple{Int,Float64}}}
    a_labels::Vector{A}
    a_lookup::Dict{Tuple{Int,A}, Int}

    # for tracking transitions
    n_a_children::Vector{Int}
    transition_rlookup::Dict{UInt64,Int}


    function ModularTree{S,A}(sz::Int=1000) where {S,A} 
        sz = min(sz, 100_000)
        return new(sizehint!(Int[], sz),
                   sizehint!(Vector{Int}[], sz),
                   sizehint!(S[], sz),
                   Dict{S, Int}(),
                   
                   sizehint!(Int[], sz),
                   sizehint!(Float64[], sz),
                   sizehint!(Vector{Tuple{Int,Float64}}[], sz),
                   sizehint!(A[], sz),
                   Dict{Tuple{Int,A}, Int}(),

                   sizehint!(Int[], sz),
                   Dict{UInt64,Int}() 
                  )
    end
end

function insert_state_node!{S,A}(tree::ModularTree{S,A}, s::S, maintain_s_lookup=true)
    push!(tree.total_n, 0)
    push!(tree.children, Int[])
    push!(tree.s_labels, s)
    snode = length(tree.total_n)
    if maintain_s_lookup
        tree.s_lookup[s] = snode
    end
    return snode
end

function insert_action_node!{S,A}(tree::ModularTree{S,A}, snode::Int, a::A, n0::Int, q0::Float64, maintain_a_lookup=true)
    push!(tree.n, n0)
    push!(tree.q, q0)
    push!(tree.a_labels, a)
    push!(tree.transitions, Vector{Tuple{Int,Float64}}[])
    sanode = length(tree.n)
    push!(tree.children[snode], sanode)
    push!(tree.n_a_children, 0)
    if maintain_a_lookup
        tree.a_lookup[(snode, a)] = sanode
    end
    return sanode
end

Base.isempty(tree::ModularTree) = isempty(tree.n) && isempty(tree.q)

immutable ModularStateNode{S,A} <: AbstractStateNode
    tree::ModularTree{S,A}
    index::Int
end

children(n::ModularStateNode) = n.tree.children[n.index]
n_children(n::ModularStateNode) = length(children(n))
isroot(n::ModularStateNode) = n.index == 1

mutable struct ModularPlanner{P<:Union{MDP,POMDP}, S, A, SE, NA, RNG} <: AbstractMCTSPlanner{P}
    solver::ModularSolver
    mdp::P
    tree::Nullable{ModularTree{S,A}}
    solved_estimate::SE
    next_action::NA
    rng::RNG
end

function ModularPlanner{S,A}(solver::ModularSolver, mdp::Union{POMDP{S,A},MDP{S,A}})
    se = convert_estimator(solver.estimate_value, solver, mdp)
    return ModularPlanner(solver,
                      mdp,
                      Nullable{ModularTree{S,A}},
                      se,
                      solver.next_action,
                      solver.rng
                     )
end

Base.srand(p::ModularPlanner, seed) = srand(p.rng, seed)

include("modules/DPWBandit.jl")
include("modules/RandomBandit.jl")
#include("modules/GPBandit.jl")
include("modules/CBTSBandit.jl")
include("modules/CBTSDPWBandit.jl")
#include("modules/QBASEBandit.jl")
include("bestpathtracker.jl")
