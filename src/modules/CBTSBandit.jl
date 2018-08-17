using GaussianProcesses

mutable struct CBTSBandit <: ModularBandit 
    enable_action_pw::Bool
    check_repeat_action::Bool
    exploration_constant::Float64
    A_max::Int
    n_proposes::Int
    n_sig::Float64
    action_dims::Int
    gp::GPE
    Xs::Vector{Matrix{Float64}}
    rs::Vector{Vector{Float64}}

    function CBTSBandit(; 
                    enable_action_pw::Bool=true,
                    check_repeat_action::Bool=true,
                    exploration_constant::Float64=1.0,
                    A_max::Int=20,
                    n_proposes::Int=50,
                    log_length_scale::Float64=0.0,
                    log_signal_sigma::Float64=0.0,
                    log_obs_noise::Float64=-1.0,
                    action_dims::Int=2,
                    n_sig::Float64=2.0,  #number of standard deviations for GP-UCB
                    )

        mean_func = MeanZero()
        kern = SE(log_length_scale, log_signal_sigma)
        gp = GP(Array{Float64,2}(action_dims,0), Float64[], mean_func, kern, log_obs_noise) 
        Xs = Matrix{Float64}[]
        rs = Vector{Float64}[]
        new(enable_action_pw, check_repeat_action, exploration_constant, A_max, n_proposes, n_sig, action_dims, gp, Xs, rs)
    end
end
Base.string(::Type{CBTSBandit}) = "CBTSBandit"

function bandit_action(p::ModularPlanner, b::CBTSBandit, snode)

    b.enable_action_pw || error("enable_action_pw=false is not supported by CBTSBandit")

    sol = p.solver
    tree = get(p.tree)
    s = tree.s_labels[snode]

    sanode = if length(tree.children[snode]) < b.A_max
        #Bayesian optimization

        a = if !isassigned(b.Xs,snode) #isempty(tree.children[snode])
            a = next_action(p.next_action, p.mdp, s, ModularStateNode(tree, snode)) 
            a
        else
            #fit to q's
            #X = hcat((tree.a_labels[i] for i in tree.children[snode])...)
            #y = [tree.q[i] for i in tree.children[snode]]

            #fit to r's
            X = b.Xs[snode]
            y = b.rs[snode]

            GaussianProcesses.fit!(b.gp, X, y)  #fit using existing actions

            #wrap sampling function for clarity
            sample_action() = next_action(p.next_action, p.mdp, s, ModularStateNode(tree, snode)) 
            test_points = hcat((sample_action() for i=1:b.n_proposes)...)  #argmax over a sampling of test points

            #test and pick the best action
            m, Σ = predict_f(b.gp, test_points)
            ucb = b.n_sig*sqrt.(Σ)
            ucbmax,imax = findmax(m + ucb)

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

function bandit_update!(p::ModularPlanner, b::CBTSBandit, snode, sanode, r, q)
    tree = get(p.tree)
    tree.q[sanode] += (q - tree.q[sanode])/tree.n[sanode]
    #resize if too small
    length(b.Xs) < snode && resize!(b.Xs, snode)
    length(b.rs) < snode && resize!(b.rs, snode)

    a = tree.a_labels[sanode]
    b.Xs[snode] = isassigned(b.Xs,snode) ? hcat(b.Xs[snode], vec(a)) : matrix(a) #might fail if a is not a vector of floats
    isassigned(b.rs,snode) || (b.rs[snode] = Float64[])
    push!(b.rs[snode], r)

    nothing
end

Base.vec(x::Float64) = [x]
matrix(x::Float64) = [x][:,:]
