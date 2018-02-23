function POMDPs.solve(solver::DSBSolver, mdp::Union{POMDP,MDP})
    S = state_type(mdp)
    A = action_type(mdp)
    se = convert_estimator(solver.estimate_value, solver, mdp)
    return DSBPlanner(solver, mdp, Nullable{DPWTree{S,A}}(), se, solver.next_action, solver.rng)
end

"""
Delete existing decision tree.
"""
function clear_tree!(p::DSBPlanner)
    p.tree = Nullable()
end

"""
Call simulate and chooses the approximate best action from the reward approximations
"""
function POMDPs.action(p::DSBPlanner, s)
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
            tree = DPWTree{S,A}(p.solver.n_iterations)
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
        tree = DPWTree{S,A}(p.solver.n_iterations)
        p.tree = Nullable(tree)
        snode = insert_state_node!(tree, s, p.solver.check_repeat_state)
    end
    start_us = CPUtime_us()
    for i = 1:p.solver.n_iterations
        simulate(p, snode, p.solver.depth) # (not 100% sure we need to make a copy of the state here)
        if CPUtime_us() - start_us >= p.solver.max_time * 1e6
            break
        end
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
    return tree.a_labels[sanode] # choose action with highest approximate value
end

action_distance{S,A}(dsb::DSBPlanner, mdp::MDP{S,A}, s::S, a1::A, a2::A) = action_distance(mdp, a1, a2) 
action_distance{S,A}(mdp::MDP{S,A}, a1::A, a2::A) = norm(a1-a2, 2) 
state_distance{S,A}(dsb::DSBPlanner, mdp::MDP{S,A}, s::S, a::A, s1::S, s2::S) = state_distance(mdp, s1, s2) 
state_distance{S,A}(mdp::MDP{S,A}, s1::S, s2::S) = norm(s1-s2, 2) 

#dsb-actions
function is_dissimilar_action{S,A}(dsb::DSBPlanner, s::S, new_action::A, actions::Vector{A}, r::Float64)
    for a in actions
        d = action_distance(dsb, dsb.mdp, s, new_action, a)
        if d <= r
            return false
        end
    end
    return true
end

#dsb-states
function is_dissimilar_state{S,A}(dsb::DSBPlanner, s::S, a::A, new_state::S, 
                                  states::Vector{S}, r::Float64)
    for sp in states
        d = state_distance(dsb, dsb.mdp, s, a, new_state, sp)
        if d <= r
            return false
        end
    end
    return true
end

"""
Return the reward for one iteration of MCTSDSB.
"""
function simulate(dsb::DSBPlanner, snode::Int, d::Int)
    S = state_type(dsb.mdp)
    A = action_type(dsb.mdp)
    sol = dsb.solver
    tree = get(dsb.tree)
    s = tree.s_labels[snode]
    if d == 0 || isterminal(dsb.mdp, s)
        return 0.0
    end

    # action progressive widening
    if dsb.solver.enable_action_pw
        if length(tree.children[snode]) <= sol.k_action*tree.total_n[snode]^sol.alpha_action # criterion for new action generation
            a = next_action(dsb.next_action, dsb.mdp, s, DPWStateNode(tree, snode)) # action generation step
            if isempty(tree.children[snode]) || is_dissimilar_action(dsb, s, a, 
                            [tree.a_labels[c] for c in tree.children[snode]], 
                             sol.r0_action/(tree.total_n[snode]^sol.lambda_action))

                if !sol.check_repeat_action || !haskey(tree.a_lookup, (snode, a))
                    n0 = init_N(sol.init_N, dsb.mdp, s, a)
                    insert_action_node!(tree, snode, a, n0, 
                                        init_Q(sol.init_Q, dsb.mdp, s, a), 
                                        sol.check_repeat_action)
                    tree.total_n[snode] += n0
                end
            end
        end
    elseif isempty(tree.children[snode])
        for a in iterator(actions(dsb.mdp, s))
            n0 = init_N(sol.init_N, dsb.mdp, s, a)
            insert_action_node!(tree, snode, a, n0,
                                init_Q(sol.init_Q, dsb.mdp, s, a),
                                false)
            tree.total_n[snode] += n0
        end
    end

    best_UCB = -Inf
    sanode = 0
    ltn = log(tree.total_n[snode])
    for child in tree.children[snode]
        n = tree.n[child]
        q = tree.q[child]
        if ltn <= 0 && n == 0
            UCB = q
        else
            c = sol.exploration_constant # for clarity
            UCB = q + c*sqrt(ltn/n)
        end
        @assert !isnan(UCB)
        @assert !isequal(UCB, -Inf)
        if UCB > best_UCB
            best_UCB = UCB
            sanode = child
        end
    end

    a = tree.a_labels[sanode]

    # state progressive widening
    new_node = false
    if tree.n_a_children[sanode] <= sol.k_state*tree.n[sanode]^sol.alpha_state
        sp, r = generate_sr(dsb.mdp, s, a, dsb.rng)

        # dsb-state
        if isempty(tree.transitions[sanode]) || is_dissimilar_state(dsb, s, a, sp, 
                        [tree.s_labels[c[1]] for c in tree.transitions[sanode]],
                         sol.r0_state/(tree.total_n[sanode]^sol.lambda_state))
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
        end
    else
        spnode, r = rand(dsb.rng, tree.transitions[sanode])
    end

    if new_node
        q = r + discount(dsb.mdp)*estimate_value(dsb.solved_estimate, dsb.mdp, sp, d-1)
    else
        q = r + discount(dsb.mdp)*simulate(dsb, spnode, d-1)
    end

    tree.n[sanode] += 1
    tree.total_n[snode] += 1

    tree.q[sanode] += (q - tree.q[sanode])/tree.n[sanode]

    return q
end
