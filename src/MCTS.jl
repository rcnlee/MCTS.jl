__precompile__()
module MCTS

using POMDPs
using POMDPToolbox

using Compat
using Blink
using CPUTime

export
    MCTSSolver,
    MCTSPlanner,
    DPWSolver,
    DPWPlanner,
    DSBSolver,
    DSBPlanner,
    ASBSolver,
    ASBPlanner,
    RSBSolver,
    RSBPlanner,
    BeliefMCTSSolver,
    AbstractMCTSPlanner,
    AbstractMCTSSolver,
    solve,
    action,
    rollout,
    StateNode,
    RandomActionGenerator,
    RolloutEstimator,
    next_action,
    clear_tree!,
    estimate_value,
    init_N,
    init_Q,
    children,
    n_children,
    isroot,
    action_distance,
    state_distance

export
    AbstractStateNode,
    StateActionStateNode,
    DPWStateActionNode,
    DPWStateNode,
    ASBStateNode,
    RSBStateNode

abstract type AbstractMCTSPlanner{P<:Union{MDP,POMDP}} <: Policy end
abstract type AbstractMCTSSolver <: Solver end
abstract type AbstractStateNode end

include("requirements_info.jl")
include("domain_knowledge.jl")
include("vanilla.jl")
include("dpw_types.jl")
include("dpw.jl")
include("dsb_types.jl")
include("dsb.jl")
include("asb_types.jl")
include("asb.jl")
include("rsb_types.jl")
include("rsb.jl")
include("action_gen.jl")
include("util.jl")
include("belief_mcts.jl")

include("visualization.jl")

end # module
