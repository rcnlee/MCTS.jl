
mutable struct DPWBandit <: ModularBandit 
    exploration_constant::Float64
    k_action::Float64
    alpha_action::Float64
    enable_action_pw::Bool
    check_repeat_action::Bool

    function DPWBandit(; 
                    exploration_constant::Float64=1.0,
                    k_action::Float64=10.0,
                    alpha_action::Float64=0.5,
                    enable_action_pw::Bool=true,
                    check_repeat_action::Bool=true
                    )

        new(exploration_constant, k_action, alpha_action, enable_action_pw, check_repeat_action)
    end
end
Base.string(::Type{DPWBandit}) = "DPWBandit"

function bandit_action(p::ModularPlanner, b::DPWBandit, snode)

    sol = p.solver
    tree = get(p.tree)
    s = tree.s_labels[snode]
    
    # for clarity
    n, N = length(tree.children[snode]), tree.total_n[snode]
    k, α = b.k_action, b.alpha_action

    # action progressive widening
    if b.enable_action_pw
        if n ≤ k*N^α  # criterion for new action generation
            a = next_action(p.next_action, p.mdp, s, ModularStateNode(tree, snode)) # action generation step
            if !b.check_repeat_action || !haskey(tree.a_lookup, (snode, a))
                n0 = init_N(sol.init_N, p.mdp, s, a)
                insert_action_node!(tree, snode, a, n0, init_Q(sol.init_Q, p.mdp, s, a), b.check_repeat_action)
                tree.total_n[snode] += n0
            end
        end
    elseif isempty(tree.children[snode])
        for a in iterator(actions(p.mdp, s))
            n0 = init_N(sol.init_N, p.mdp, s, a)
            insert_action_node!(tree, snode, a, n0, init_Q(sol.init_Q, p.mdp, s, a), false)
            tree.total_n[snode] += n0
        end
    end

    best_UCB = -Inf
    sanode = 0
    ltn = log(tree.total_n[snode])
    for child in tree.children[snode]
        n = tree.n[child]
        q = tree.q[child]
        c = b.exploration_constant # for clarity
        if (ltn <= 0 && n == 0) || c == 0.0
            UCB = q
        else
            UCB = q + c*sqrt(ltn/n)
        end
        @assert !isnan(UCB) "UCB was NaN (q=$q, c=$c, ltn=$ltn, n=$n)"
        @assert !isequal(UCB, -Inf)
        if UCB > best_UCB
            best_UCB = UCB
            sanode = child
        end
    end
    sanode
end

function bandit_update!(p::ModularPlanner, b::DPWBandit, sanode, r)
    tree = get(p.tree)
    tree.q[sanode] += (r - tree.q[sanode])/tree.n[sanode]
    nothing
end