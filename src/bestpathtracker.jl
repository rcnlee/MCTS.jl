export TrackedRolloutSimulator, BestPathTracker, BPTrackerPolicy


using POMDPs, POMDPToolbox

mutable struct BestPathTracker
    best_states::Vector
    best_actions::Vector
    best_r_total::Float64
    cur_states::Vector
    cur_actions::Vector
    cur_r_total::Float64
    discount::Float64
    cur_discount::Float64

    BestPathTracker(; discount::Float64=1.0) = new([], [], -Inf, [], [], 0.0, discount, 1.0)
end

function reset_best!(t::BestPathTracker)
    empty!(t.best_states)
    empty!(t.best_actions)
    t.best_r_total = -Inf
    t
end
function reset_current!(t::BestPathTracker)
    empty!(t.cur_states)
    empty!(t.cur_actions)
    t.cur_r_total = 0.0
    t
end
function reset!(t::BestPathTracker)
    reset_best!(t)
    reset_current!(t)
    t
end

function update!(t::BestPathTracker, s, a, r::Float64)
    push!(t.cur_states, s)
    push!(t.cur_actions, a)
    t.cur_r_total += t.cur_discount*r 
    t.cur_discount *= t.discount
    t
end

function complete_current_path!(t::BestPathTracker)
    if t.cur_r_total > t.best_r_total
        t.best_r_total = t.cur_r_total
        resize_copy!(t.best_states, t.cur_states)
        resize_copy!(t.best_actions, t.cur_actions)
    end
    reset_current!(t)
    t
end

function POMDPs.simulate(t::BestPathTracker, p::ModularPlanner, mdp, s; rng::AbstractRNG=Base.GLOBAL_RNG)
    hr = HistoryRecorder(; rng=rng)
    policy = BPTrackerPolicy(p, t)
    h = simulate(hr, mdp, policy, s) 
    h
end

function resize_copy!{T}(dst::AbstractVector{T}, src::AbstractVector{T})
    resize!(dst, length(src))
    copy!(dst, src)
    dst
end

mutable struct BPTrackerPolicy <: Policy
    t::BestPathTracker
    index::Int

    function BPTrackerPolicy(t::BestPathTracker)
        new(t, 1)
    end
end

function POMDPs.action(policy::BPTrackerPolicy, s)
    if policy.index > length(policy.t.best_actions) 
        return policy.t.best_actions[end]
    else
        a = policy.t.best_actions[policy.index] 
        policy.index += 1
        return a
    end
end

mutable struct TrackedRolloutSimulator <: Simulator
    rng::AbstractRNG
    policy::Policy

    # optional: if these are null, they will be ignored
    max_steps::Nullable{Int}
    eps::Nullable{Float64}
end
function TrackedRolloutSimulator(rng::AbstractRNG, policy::Policy, d::Int=typemax(Int)) 
    TrackedRolloutSimulator(rng, policy, Nullable{Int}(d), Nullable{Float64}())
end
function estimate_value(estimator, mdp, state, depth, best_path) 
    estimate_value(estimator, mdp, state, depth) # default back to original
end
function estimate_value(sim::TrackedRolloutSimulator, mdp::MDP, initial_state, depth::Int, 
                        best_path::Nullable{BestPathTracker})

    eps = get(sim.eps, 0.0)
    max_steps = get(sim.max_steps, typemax(Int))

    s = initial_state

    disc = 1.0
    r_total = 0.0
    step = 1

    while disc > eps && !isterminal(mdp, s) && step <= max_steps
        a = action(sim.policy, s)

        sp, r = generate_sr(mdp, s, a, sim.rng)
        !isnull(best_path) && update!(get(best_path), s, a, r)  #rlee: added

        r_total += disc*r

        s = sp

        disc *= discount(mdp)
        step += 1
    end

    return r_total
end
