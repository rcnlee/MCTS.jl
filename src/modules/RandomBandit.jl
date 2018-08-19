using Observers

mutable struct RandomBandit <: ModularBandit 
    enable_action_pw::Bool
    check_repeat_action::Bool
    observer::Union{Observer,Void}

    function RandomBandit(; 
                    enable_action_pw::Bool=true,
                    check_repeat_action::Bool=true,
                    observer::Union{Observer,Void}=nothing
                    )

        new(enable_action_pw, check_repeat_action, observer)
    end
end
Base.string(::Type{RandomBandit}) = "RandomBandit"

function bandit_action(p::ModularPlanner, b::RandomBandit, snode)

    b.enable_action_pw || error("enable_action_pw=false is not supported by RandomBandit")

    sol = p.solver
    tree = get(p.tree)
    s = tree.s_labels[snode]

    a = next_action(p.next_action, p.mdp, s, ModularStateNode(tree, snode)) # action generation step
    if !b.check_repeat_action || !haskey(tree.a_lookup, (snode, a))
        n0 = init_N(sol.init_N, p.mdp, s, a)
        insert_action_node!(tree, snode, a, n0, init_Q(sol.init_Q, p.mdp, s, a), b.check_repeat_action)
        tree.total_n[snode] += n0
    end

    sanode = tree.a_lookup[(snode,a)]
    sanode
end

function bandit_update!(p::ModularPlanner, b::RandomBandit, snode, sanode, r, q)
    tree = get(p.tree)
    tree.q[sanode] += (q - tree.q[sanode])/tree.n[sanode]

    notify_observer!(b.observer, b; planner=p, snode=snode, sanode=sanode, r=r, q=q)

    nothing
end
