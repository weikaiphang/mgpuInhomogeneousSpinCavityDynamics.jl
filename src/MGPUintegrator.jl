Base.@kwdef struct SolverOptions{T}
    integrator::Symbol = :ck45
    save_mode::Symbol = :tstops
    reltol::T = T(1e-8)
    abstol::T = T(1e-8)
    dtmax::T = T(Inf)
    dtmin::T = T(0)
    dt0::T = T(0)
    maxiters::Int = 10_000_000
    verbose::Bool = true
    progress_every::Int = 0
end

mutable struct SolveStats
    nsteps::Int
    naccept::Int
    nreject::Int
    nrhs::Int
    dtmin_used::Float64
    dtmax_used::Float64
    elapsed::Float64
end

SolveStats() = SolveStats(0, 0, 0, 0, Inf, 0.0, 0.0)



const TSIT5_U, TSIT5_TMP = 1, 2
const TSIT5_K = (3, 4, 5, 6, 7, 8, 9)
const TSIT5_INTERP = TSIT5_K[2]

const CK_UPREV, CK_STAGE, CK_ACC, CK_K, CK_ERR = 1, 2, 3, 4, 5


function initial_step(prob::MGPUProblem{T}, tab, t0::Real, tend::Real,
                      iu::Int, if0::Int, opts::SolverOptions{T}) where {T}
    if opts.dt0 > 0
        return T(opts.dt0)
    end

    span = T(abs(tend - t0))
    d0 = diff_norm(prob, iu, 0, iu)
    d1 = diff_norm(prob, if0, 0, iu)

    dt = (d0 < T(1e-5) || d1 < T(1e-5)) ? T(1e-6) * span : T(0.01) * (d0 / d1)
    dt = min(dt, span, opts.dtmax)
    dt = max(dt, T(1e-12) * span)
    return dt
end



function solve_mgpu!(prob::MGPUProblem{T}, t0::Real, tend::Real,
                     tsave::AbstractVector, on_save::F,
                     opts::SolverOptions{T}) where {T,F}
    tab = build_tableau(opts.integrator, T)
    ctrl = PIController(T, alg_order(tab))
    stats = SolveStats()

    nsave = length(tsave)
    isave = 1

    t = T(t0)
    tfinal = T(tend)


    if nsave >= 1 && abs(tsave[1] - t0) <= eps(T) * max(one(T), abs(T(t0)))
        on_save(1, T(t0), TSIT5_U)
        isave = 2
    end

    interp = opts.integrator === :tsit5 ? Tsit5Interp(T) : nothing
    if opts.save_mode === :interpolate && opts.integrator !== :tsit5
        error("save_mode = :interpolate requires integrator = :tsit5 " *
              "(the low-storage method has no dense output).")
    end


    if opts.integrator === :tsit5
        rhs!(prob, TSIT5_U, TSIT5_K[1], t)
        dt = initial_step(prob, tab, t, tfinal, TSIT5_U, TSIT5_K[1], opts)
    else
        rhs!(prob, CK_UPREV, CK_K, t)
        dt = initial_step(prob, tab, t, tfinal, CK_UPREV, CK_K, opts)
    end

    dt_free = dt
    tstart_wall = time_ns()

    while t < tfinal && stats.nsteps < opts.maxiters
        dt = min(dt_free, opts.dtmax, tfinal - t)
        forced = false
        t_target = zero(T)

        if opts.save_mode === :tstops && isave <= nsave
            tnext = T(tsave[isave])
            if t + dt >= tnext - 100 * eps(T) * max(one(T), abs(tnext))
                dt = tnext - t
                forced = true
                t_target = tnext
            end
        end

        dt <= 0 && break
        stats.nsteps += 1
        stats.dtmin_used = min(stats.dtmin_used, Float64(dt))
        stats.dtmax_used = max(stats.dtmax_used, Float64(dt))

        EEst = opts.integrator === :tsit5 ?
               tsit5_step!(prob, tab, t, dt) :
               ck45_step!(prob, tab, t, dt)

        if EEst <= one(T)
            stats.naccept += 1
            tprev = t
            t = forced ? t_target : t + dt

            if opts.integrator === :tsit5

                if opts.save_mode === :interpolate
                    while isave <= nsave &&
                          T(tsave[isave]) <= t + 100 * eps(T) * max(one(T), abs(t))
                        Θ = min(one(T), max(zero(T), (T(tsave[isave]) - tprev) / dt))
                        w = interp_weights(interp, Θ)



                        combine!(prob, TSIT5_INTERP, TSIT5_U, TSIT5_K, w, dt;
                                 n = :save)
                        on_save(isave, T(tsave[isave]), TSIT5_INTERP)
                        isave += 1
                    end
                end

                swap_registers!(prob, TSIT5_U, TSIT5_TMP)
                swap_registers!(prob, TSIT5_K[1], TSIT5_K[7])

                if opts.save_mode === :tstops && forced && isave <= nsave &&
                   T(tsave[isave]) == t_target
                    on_save(isave, t, TSIT5_U)
                    isave += 1
                end
            else
                swap_registers!(prob, CK_UPREV, CK_ACC)
                if forced && isave <= nsave && T(tsave[isave]) == t_target
                    on_save(isave, t, CK_UPREV)
                    isave += 1
                end
            end

            accept_step!(ctrl, EEst)
            q, _ = controller_factors(ctrl, EEst)
            dt_new = dt / q
            dt_free = forced ? max(dt_free, dt_new) : dt_new
        else
            stats.nreject += 1
            _, q11 = controller_factors(ctrl, EEst)
            dt_free = dt / min(inv(ctrl.qmin), q11 / ctrl.gamma)
            if dt_free < opts.dtmin || !isfinite(dt_free)
                error("Step size underflow at t = $t (dt = $dt_free, EEst = $EEst).")
            end
        end

        if opts.progress_every > 0 && stats.nsteps % opts.progress_every == 0
            @printf("  t = %.6e / %.6e   dt = %.3e   steps = %d (%d rejected)\n",
                    t, tfinal, dt_free, stats.nsteps, stats.nreject)
            flush(stdout)
        end
    end

    if stats.nsteps >= opts.maxiters
        @warn "Reached maxiters before the end of the time span." t tfinal stats.nsteps
    end

    stats.elapsed = (time_ns() - tstart_wall) / 1e9
    stats.nrhs = prob.nrhs[]
    return stats
end


function tsit5_step!(prob::MGPUProblem{T}, tab::Tsit5Tableau{T}, t::T, dt::T) where {T}
    k1, k2, k3, k4, k5, k6, k7 = TSIT5_K
    u, tmp = TSIT5_U, TSIT5_TMP

    combine!(prob, tmp, u, (k1,), (tab.a21,), dt)
    rhs!(prob, tmp, k2, t + tab.c2 * dt)

    combine!(prob, tmp, u, (k1, k2), (tab.a31, tab.a32), dt)
    rhs!(prob, tmp, k3, t + tab.c3 * dt)

    combine!(prob, tmp, u, (k1, k2, k3), (tab.a41, tab.a42, tab.a43), dt)
    rhs!(prob, tmp, k4, t + tab.c4 * dt)

    combine!(prob, tmp, u, (k1, k2, k3, k4),
             (tab.a51, tab.a52, tab.a53, tab.a54), dt)
    rhs!(prob, tmp, k5, t + tab.c5 * dt)

    combine!(prob, tmp, u, (k1, k2, k3, k4, k5),
             (tab.a61, tab.a62, tab.a63, tab.a64, tab.a65), dt)
    rhs!(prob, tmp, k6, t + tab.c6 * dt)

    combine!(prob, tmp, u, (k1, k2, k3, k4, k5, k6),
             (tab.a71, tab.a72, tab.a73, tab.a74, tab.a75, tab.a76), dt)
    rhs!(prob, tmp, k7, t + dt)

    return error_norm(prob, u, tmp, TSIT5_K,
                      (tab.bt1, tab.bt2, tab.bt3, tab.bt4, tab.bt5, tab.bt6, tab.bt7),
                      dt)
end


function ck45_step!(prob::MGPUProblem{T}, tab::CK45Tableau{T}, t::T, dt::T) where {T}
    ns = 5
    for i in 1:ns
        first = i == 1


        if !first
            rhs!(prob, CK_STAGE, CK_K, t + tab.c[i] * dt)
        end
        cA = i < ns ? tab.A[i] : zero(T)
        lowstorage!(prob, dt, cA, tab.b[i], tab.bt[i], i < ns, first)
    end

    EEst = error_norm(prob, CK_UPREV, CK_ACC, (CK_ERR,), (one(T),), one(T))


    if EEst <= one(T)
        rhs!(prob, CK_ACC, CK_K, t + dt)
    else
        rhs!(prob, CK_UPREV, CK_K, t)
    end
    return EEst
end

function lowstorage!(prob::MGPUProblem{T}, dt::T, cA::T, cB::T, cE::T,
                     write_stage::Bool, first::Bool) where {T}
    each_shard(prob.shards, prob.exec) do s
        base = first ? s.regs[CK_UPREV] : s.regs[CK_ACC]
        @cuda threads=256 blocks=s.nblocks_vec stream=s.stream lowstorage_kernel!(
            s.regs[CK_STAGE], s.regs[CK_ACC], s.regs[CK_ERR], s.regs[CK_K],
            base, dt, cA, cB, cE, s.n, write_stage, first)
    end
    return nothing
end
