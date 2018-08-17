function POMDPs.solve(solver::ModularSolver, mdp::Union{POMDP,MDP})
    S = state_type(mdp)
    A = action_type(mdp)
    se = convert_estimator(solver.estimate_value, solver, mdp)
    return ModularPlanner(solver, mdp, Nullable{ModularTree{S,A}}(), se, solver.next_action, solver.rng)
end

"""
Delete existing decision tree.
"""
function clear_tree!(p::ModularPlanner)
    p.tree = Nullable()
end

"""
Construct an Modular MCTS tree and choose the best action.
"""
POMDPs.action(p::ModularPlanner, s) = first(action_info(p, s))

"""
Construct an Modular MCTS tree and choose the best action. Also output some information.
"""
function POMDPToolbox.action_info(p::ModularPlanner, s; tree_in_info=false)
    local a::action_type(p.mdp)
    info = Dict{Symbol, Any}()
    try
        if isterminal(p.mdp, s)
            error("""
                  MCTS cannot handle terminal states. action was called with
                  s = $s
                  """)
        end

        S = state_type(p.mdp)
        A = action_type(p.mdp)
        if p.solver.keep_tree
            if isnull(p.tree)
                tree = ModularTree{S,A}(p.solver.n_iterations)
                p.tree = Nullable(tree)
            else
                tree = get(p.tree)
            end
            if haskey(tree.s_lookup, s)
                snode = tree.s_lookup[s]
            else
                snode = insert_state_node!(tree, s, true)
            end
        else
            tree = ModularTree{S,A}(p.solver.n_iterations)
            p.tree = Nullable(tree)
            snode = insert_state_node!(tree, s, p.solver.check_repeat_state)
        end

        i = 0
        start_us = CPUtime_us()
        best_path = p.solver.track_best_path ? Nullable{BestPathTracker}(BestPathTracker()) : Nullable{BestPathTracker}() 
        for i = 1:p.solver.n_iterations
            simulate(p, snode, p.solver.depth, best_path) 
            !isnull(best_path) && complete_current_path!(get(best_path))

            if CPUtime_us() - start_us >= p.solver.max_time * 1e6
                break
            end
        end
        info[:search_time_us] = CPUtime_us() - start_us
        info[:tree_queries] = i
        if p.solver.tree_in_info || tree_in_info
            info[:tree] = tree
        end
        if !isnull(best_path) 
            info[:best_path] = get(best_path)
        end
        
        best_Q = -Inf
        sanode = 0
        for child in tree.children[snode]
            if tree.q[child] > best_Q
                best_Q = tree.q[child]
                sanode = child
            end
        end
        # XXX some publications say to choose action that has been visited the most
        a = tree.a_labels[sanode] # choose action with highest approximate value
    catch ex
        a = convert(action_type(p.mdp), default_action(p.solver.default_action, p.mdp, s, ex))
        info[:exception] = ex
    end

    return a, info
end


"""
Return the reward for one iteration of Modular MCTS.
"""
function simulate(p::ModularPlanner, snode::Int, d::Int, best_path::Nullable{BestPathTracker}=Nullable{BestPathTracker}())
    S = state_type(p.mdp)
    A = action_type(p.mdp)
    sol = p.solver
    tree = get(p.tree)
    s = tree.s_labels[snode]
    if d == 0 || isterminal(p.mdp, s)
        return 0.0
    end

    sanode = bandit_action(p, p.solver.bandit, snode)
    a = tree.a_labels[sanode]

    # state progressive widening
    new_node = false
    if tree.n_a_children[sanode] <= sol.k_state*tree.n[sanode]^sol.alpha_state
        sp, r = generate_sr(p.mdp, s, a, p.rng)
        !isnull(best_path) && update!(get(best_path), s, a, r)

        spnode = sol.check_repeat_state ? get(tree.s_lookup, sp, 0) : 0

        if spnode == 0 # there was not a state node for sp already in the tree
            spnode = insert_state_node!(tree, sp, sol.keep_tree || sol.check_repeat_state)
            new_node = true
        end
        push!(tree.transitions[sanode], (spnode, r))

        if !sol.check_repeat_state 
            tree.n_a_children[sanode] += 1
        elseif !((sanode,spnode) in tree.unique_transitions)
            push!(tree.unique_transitions, (sanode,spnode))
            tree.n_a_children[sanode] += 1
        end
    else
        spnode, r = rand(p.rng, tree.transitions[sanode])
    end

    if new_node
        q = r + discount(p.mdp)*estimate_value(p.solved_estimate, p.mdp, sp, d-1, best_path)
    else
        q = r + discount(p.mdp)*simulate(p, spnode, d-1, best_path)
    end

    tree.n[sanode] += 1
    tree.total_n[snode] += 1

    bandit_update!(p, p.solver.bandit, snode, sanode, r, q)

    return q
end
