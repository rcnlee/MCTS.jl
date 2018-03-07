function POMDPs.solve(solver::ASBSolver, mdp::Union{POMDP,MDP})
    S = state_type(mdp)
    A = action_type(mdp)
    se = convert_estimator(solver.estimate_value, solver, mdp)
    return ASBPlanner(solver, mdp, Nullable{ASBTree{S,A}}(), se, solver.next_action, solver.rng)
end

"""
Delete existing decision tree.
"""
function clear_tree!(p::ASBPlanner)
    p.tree = Nullable()
end

"""
Call simulate and chooses the approximate best action from the reward approximations
"""
function POMDPs.action(p::ASBPlanner, s)
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
            tree = ASBTree{S,A}(p.solver.n_iterations)
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
        tree = ASBTree{S,A}(p.solver.n_iterations)
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

action_distance{S,A}(asb::ASBPlanner, mdp::MDP{S,A}, s::S, a1::A, a2::A) = action_distance(mdp, a1, a2) 
action_distance{S,A}(mdp::MDP{S,A}, a1::A, a2::A) = norm(a1-a2, 2) 
state_distance{S,A}(asb::ASBPlanner, mdp::MDP{S,A}, s::S, a::A, s1::S, s2::S) = state_distance(mdp, s1, s2) 
state_distance{S,A}(mdp::MDP{S,A}, s1::S, s2::S) = norm(s1-s2, 2) 

function nearest_neighbor{S,A}(asb::ASBPlanner, s::S, snode, a::A)
    tree = get(asb.tree)
    nn, dist = nothing, Inf
    for x in tree.children[snode]
        d = action_distance(asb, asb.mdp, s, a, tree.a_labels[x])
        rad = tree.a_radius[x]
        if d < rad && d < dist
            nn,dist = x, d
        end
    end
    nn,dist
end
function nearest_neighbor{S,A}(asb::ASBPlanner, s::S, a::A, sanode, sp::S)
    tree = get(asb.tree)
    nn, dist = nothing, Inf
    for x in tree.transitions[sanode] #this is very inefficient, may contain repetitions!
        cnode, r = x
        d = state_distance(asb, asb.mdp, s, a, sp, tree.s_labels[cnode]) 
        rad = tree.sp_radius[(sanode,cnode)]
        if d < rad && d < dist
            nn, dist = x, d
        end
    end
    nn,dist
end

"""
Return the reward for one iteration of MCTSASB.
"""
function simulate(asb::ASBPlanner, snode::Int, d::Int)
    S = state_type(asb.mdp)
    A = action_type(asb.mdp)
    sol = asb.solver
    tree = get(asb.tree)
    s = tree.s_labels[snode]
    if d == 0 || isterminal(asb.mdp, s)
        return 0.0
    end

    # action progressive widening
    if asb.solver.enable_action_pw
        if length(tree.children[snode]) <= sol.k_action*tree.total_n[snode]^sol.alpha_action # criterion for new action generation
            a = next_action(asb.next_action, asb.mdp, s, ASBStateNode(tree, snode)) # action generation step
            if !sol.check_repeat_action || !haskey(tree.a_lookup, (snode, a))
                nn,dist = nearest_neighbor(asb, s, snode, a)
                if !isinf(dist)  #has neighbor
                    tree.a_radius[nn] *= sol.lambda_action #shrink ball
                else
                    n0 = init_N(sol.init_N, asb.mdp, s, a)
                    insert_action_node!(tree, snode, a, n0, 
                                        init_Q(sol.init_Q, asb.mdp, s, a), 
                                        sol.r0_action,
                                        sol.check_repeat_action)
                    tree.total_n[snode] += n0
                end
            end #else: discard
        end
    elseif isempty(tree.children[snode])
        for a in iterator(actions(asb.mdp, s))
            n0 = init_N(sol.init_N, asb.mdp, s, a)
            insert_action_node!(tree, snode, a, n0,
                                init_Q(sol.init_Q, asb.mdp, s, a),
                                sol.r0_action,
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
        sp, r = generate_sr(asb.mdp, s, a, asb.rng)

        spnode = sol.check_repeat_state ? get(tree.s_lookup, sp, 0) : 0
        prev_seen = (spnode != 0) && ((sanode,spnode) in tree.unique_transitions) #previously-seen transition
        if !prev_seen
            nn,dist = nearest_neighbor(asb, s, a, sanode, sp) 
            if !isinf(dist) #similar to existing, snap to neighbor
                spnode,r = nn 
                sp = tree.s_labels[spnode] #for downstream use
                tree.sp_radius[(sanode,spnode)] *= sol.lambda_state
            else
                if spnode == 0 # there was not a state node for sp already in the tree
                    spnode = insert_state_node!(tree, sp, sol.keep_tree || sol.check_repeat_state)
                    new_node = true
                end
                if sol.check_repeat_state
                    push!(tree.unique_transitions, (sanode,spnode))
                    tree.sp_radius[(sanode,spnode)] = sol.r0_state
                end
                tree.n_a_children[sanode] += 1
            end
        end
        push!(tree.transitions[sanode], (spnode, r))
    else
        spnode, r = rand(asb.rng, tree.transitions[sanode])
    end

    if new_node
        q = r + discount(asb.mdp)*estimate_value(asb.solved_estimate, asb.mdp, sp, d-1)
    else
        q = r + discount(asb.mdp)*simulate(asb, spnode, d-1)
    end

    tree.n[sanode] += 1
    tree.total_n[snode] += 1

    tree.q[sanode] += (q - tree.q[sanode])/tree.n[sanode]

    return q
end
