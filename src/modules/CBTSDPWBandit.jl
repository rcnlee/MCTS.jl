using GaussianProcesses
using Observers

mutable struct CBTSDPWBandit <: ModularBandit 
    enable_action_pw::Bool
    check_repeat_action::Bool
    exploration_constant::Float64
    k_action::Float64
    alpha_action::Float64
    n_proposes::Int
    n_sig::Float64
    fit_qs::Bool  #true=fit qs, false=fit rs
    action_dims::Int
    gp::GPE
    observer::Union{Observer,Void}

    function CBTSDPWBandit(; 
                    enable_action_pw::Bool=true,
                    check_repeat_action::Bool=true,
                    exploration_constant::Float64=1.0,
                    k_action::Float64=10.0,
                    alpha_action::Float64=0.5,
                    n_proposes::Int=50,
                    log_length_scale::Float64=0.0,
                    log_signal_sigma::Float64=0.0,
                    log_obs_noise::Float64=-1.0,
                    fit_qs::Bool=false,
                    action_dims::Int=2,
                    n_sig::Float64=2.0,  #number of standard deviations for GP-UCB
                    observer::Union{Observer,Void}=nothing
                    )

        mean_func = MeanZero()
        kern = SE(log_length_scale, log_signal_sigma)
        gp = GP(Array{Float64,2}(action_dims,0), Float64[], mean_func, kern, log_obs_noise) 
        new(enable_action_pw, check_repeat_action, exploration_constant, k_action, alpha_action, n_proposes, n_sig, fit_qs, action_dims, gp, observer)
    end
end
Base.string(::Type{CBTSDPWBandit}) = "CBTSDPWBandit"

function bandit_action(p::ModularPlanner, b::CBTSDPWBandit, snode)

    b.enable_action_pw || error("enable_action_pw=false is not supported by CBTSDPWBandit")

    sol = p.solver
    tree = get(p.tree)
    s = tree.s_labels[snode]

    # for clarity
    n, N = length(tree.children[snode]), tree.total_n[snode]
    k, α = b.k_action, b.alpha_action

    sanode = if n ≤ k*N^α  # criterion for new action generation
        #Generate action using Bayesian optimization

        a = if isempty(tree.children[snode]) #don't bother fitting if no existing children
            a = next_action(p.next_action, p.mdp, s, ModularStateNode(tree, snode)) 
            a
        else
            #fit to q's
            X = hcat((tree.a_labels[i] for i in tree.children[snode])...)
            y = [tree.q[i] for i in tree.children[snode]]

            GaussianProcesses.fit!(b.gp, X, y)  #fit using existing actions

            #wrap sampling function for clarity
            sample_action() = next_action(p.next_action, p.mdp, s, ModularStateNode(tree, snode)) 
            test_points = hcat((sample_action() for i=1:b.n_proposes)...)  #argmax over a sampling of test points

            #test and pick the best action
            m, Σ = predict_f(b.gp, test_points)
            ucb = b.n_sig*sqrt.(Σ)
            ucbmax,imax = findmax(m + ucb)  #do I need random tie-breaking?

            a = b.action_dims==1 ? test_points[imax] : test_points[:,imax]
            a
        end
    
        if !b.check_repeat_action || !haskey(tree.a_lookup, (snode, a))
            n0 = init_N(sol.init_N, p.mdp, s, a)
            insert_action_node!(tree, snode, a, n0, init_Q(sol.init_Q, p.mdp, s, a), b.check_repeat_action)
            tree.total_n[snode] += n0
        end
        sanode = tree.a_lookup[(snode,a)]
        sanode
    else
        #UCT criteria
        
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
    return sanode
end

function bandit_update!(p::ModularPlanner, b::CBTSDPWBandit, snode, sanode, r, q)
    tree = get(p.tree)
    tree.q[sanode] += (q - tree.q[sanode])/tree.n[sanode]

    notify_observer!(b.observer, b; planner=p, snode=snode, sanode=sanode, r=r, q=q)

    nothing
end
