function POMDPs.solve(solver::RSBSolver, mdp::Union{POMDP,MDP})
    S = state_type(mdp)
    A = action_type(mdp)
    se = convert_estimator(solver.estimate_value, solver, mdp)
    return RSBPlanner(solver, mdp, Nullable{RSBTree{S,A}}(), se, solver.next_action, solver.rng)
end

"""
Delete existing decision tree.
"""
function clear_tree!(p::RSBPlanner)
    p.tree = Nullable()
end

"""
Call simulate and chooses the approximate best action from the reward approximations
"""
function POMDPs.action(p::RSBPlanner, s)
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
            tree = RSBTree{S,A}(p.solver.n_iterations)
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
        tree = RSBTree{S,A}(p.solver.n_iterations)
        p.tree = Nullable(tree)
        snode = insert_state_node!(tree, s, p.solver.check_repeat_state)
    end
    start_us = CPUtime_us()
    for i = 1:p.solver.n_iterations
        q = simulate(p, snode, p.solver.depth) # (not 100% sure we need to make a copy of the state here)
        haskey(p.solver.listeners,:return) && notify_listener(p.solver.listeners[:return], p, i, q)
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

action_distance{S,A}(rsb::RSBPlanner, mdp::MDP{S,A}, s::S, a1::A, a2::A) = action_distance(mdp, a1, a2) 
#action_distance{S,A}(mdp::MDP{S,A}, a1::A, a2::A) = norm(a1-a2, 2)  #duplicate
state_distance{S,A}(rsb::RSBPlanner, mdp::MDP{S,A}, s::S, a::A, s1::S, s2::S) = state_distance(mdp, s1, s2) 
#state_distance{S,A}(mdp::MDP{S,A}, s1::S, s2::S) = norm(s1-s2, 2)  #duplicate 

function nearest_neighbor{S,A}(rsb::RSBPlanner, s::S, snode, a::A)
    tree = get(rsb.tree)
    nn, dist = nothing, Inf
    for x in tree.children[snode]
        d = action_distance(rsb, rsb.mdp, s, a, tree.a_labels[x])
        rad = tree.a_radius[x]
        if d < rad && d < dist
            nn,dist = x, d
        end
    end
    nn,dist
end
function nearest_neighbor{S,A}(rsb::RSBPlanner, s::S, a::A, sanode, sp::S)
    tree = get(rsb.tree)
    nn, dist = nothing, Inf
    for x in tree.transitions[sanode] 
        cnode, r = x
        d = state_distance(rsb, rsb.mdp, s, a, sp, tree.s_labels[cnode]) 
        if d < dist
            nn, dist = x, d
        end
    end
    nn,dist
end

"""
Return the reward for one iteration of MCTSRSB.
"""
function simulate(rsb::RSBPlanner, snode::Int, d::Int)
    S = state_type(rsb.mdp)
    A = action_type(rsb.mdp)
    sol = rsb.solver
    tree = get(rsb.tree)
    s = tree.s_labels[snode]
    if d == 0 || isterminal(rsb.mdp, s)
        return 0.0
    end

    # action progressive widening
    a = next_action(rsb.next_action, rsb.mdp, s, RSBStateNode(tree, snode)) # action generation step
    if !sol.check_repeat_action || !haskey(tree.a_lookup, (snode, a))  #not an existing action
        actions, n = tree.children[snode], tree.total_n[snode]
        qs = tree.q[actions]
        inds = sortperm(qs, rev=true)
        #tree.a_radius[actions[inds]] .= sol.r0_action/n^sol.lambda_action*([(x-1) for x=1:length(actions)]./length(actions))
        tree.a_radius[actions[inds]] .= sol.r0_action/n^sol.lambda_action*[(x-1) for x=1:length(actions)]
        nn,dist = nearest_neighbor(rsb, s, snode, a)
        if isinf(dist) # no neighbor, add it 
            n0 = init_N(sol.init_N, rsb.mdp, s, a)
            insert_action_node!(tree, snode, a, n0, init_Q(sol.init_Q, rsb.mdp, s, a), sol.r0_action/n^sol.lambda_action, 
                                sol.check_repeat_action)
            tree.total_n[snode] += n0
        end
    end #else: discard

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
    sp, r = generate_sr(rsb.mdp, s, a, rsb.rng)

    spnode = sol.check_repeat_state ? get(tree.s_lookup, sp, 0) : 0
    prev_seen = (spnode != 0) && ((sanode,spnode) in tree.unique_transitions) #previously-seen transition
    if !prev_seen
        nn,dist = nearest_neighbor(rsb, s, a, sanode, sp) 
        if isinf(dist) || (dist > sol.r0_state/(tree.n[sanode]^sol.lambda_state)) #very different, add it
            if spnode == 0 # there was not a state node for sp already in the tree
                spnode = insert_state_node!(tree, sp, sol.keep_tree || sol.check_repeat_state)
                new_node = true
            end
            if sol.check_repeat_state
                push!(tree.unique_transitions, (sanode,spnode))
                push!(tree.transitions[sanode], (spnode, r)) #consider consolidating transitions and unique_transitions
            end
            tree.n_a_children[sanode] += 1
        else #similar to existing, snap to neighbor
            spnode,r = nn 
            sp = tree.s_labels[spnode] #for downstream use
        end
    end
    haskey(sol.listeners,:sim) && notify_listener(sol.listeners[:sim], rsb, s, a, sp, r, snode, sanode, spnode, d)

    if new_node
        q = r + discount(rsb.mdp)*estimate_value(rsb.solved_estimate, rsb.mdp, sp, d-1)
    else
        q = r + discount(rsb.mdp)*simulate(rsb, spnode, d-1)
    end

    tree.n[sanode] += 1
    tree.total_n[snode] += 1
    tree.q[sanode] += (q - tree.q[sanode])/tree.n[sanode]

    return q
end

notify_listener(::Any, ::RSBPlanner, iter, q) = nothing
notify_listener(::Any, ::RSBPlanner, s, a, sp, r, snode, sanode, spnode, d) = nothing
