# locate run_local_adam. Right before the for epoch in 1:num_epochs loop begins, add the tracking variable:
adam_t0 = adam.t
    lr = learning_rate
    just_reverted = false
    
    prev_dyn_w_time = base_w_time # ADD THIS LINE: Tracks weight for apples-to-apples shifting

    for epoch in 1:num_epochs
        t_wall = time()


#Delete the entire _reconstitute_epoch_cost function and the tmax_frac_sq/power_mean extractions inside the loop, and replace them with the Moving Target shift:
# Override w_time whenever dyn_w_time could differ from base_w_time:
epoch_cost_kwargs = merge(cost_kwargs, (w_time=dyn_w_time,))
# --------------------------------------------------------
# THE MOVING TARGET FIX
# Shift historical costs so they are evaluated under the 
# CURRENT epoch's w_time rules. This allows Adam to accept
# duration-expanding steps during hop 0, while still 
# naturally tightening the belt in later hops as w_time grows.
# --------------------------------------------------------
if epoch > 1 && dyn_w_time != prev_dyn_w_time
    w_diff = dyn_w_time - prev_dyn_w_time
    best_cost += w_diff * (best_dur / pulse.T_max)
    
    # last_good_aux = (cost, inv, sil, dur, coh, famp)
    if isfinite(last_good_aux[1])
        shifted_last_cost = last_good_aux[1] + w_diff * (last_good_aux[4] / pulse.T_max)
        last_good_aux = (shifted_last_cost, last_good_aux[2], last_good_aux[3], 
                         last_good_aux[4], last_good_aux[5], last_good_aux[6])
    end
end
prev_dyn_w_time = dyn_w_time
# --------------------------------------------------------

#Remove the calls to _reconstitute_epoch_cost() inside the gradient try-catch blocks. Let cost equal the raw dynamic cost directly off the tape:
if just_reverted
    grad = last_good_grad
    cost, inv_, sil_, dur_, coh_, famp_ = last_good_aux
    adam.m .= adam_m0
    adam.v .= adam_v0
    adam.t = adam_t0
elseif grad_mode === :adjoint
    grad, cost, inv_, sil_, dur_, coh_, famp_ = pulse_cost_grad_adjoint(
        u, pulse, d; epoch_cost_kwargs..., compute=compute, solve_kwargs...,
    )
elseif threaded_grad
    grad, cost, inv_, sil_, dur_, coh_, famp_ = _pulse_cost_grad_threaded(
        u, pulse, d; epoch_cost_kwargs..., compute=compute, solve_kwargs...,
    )
else
    function cost_only(uu)
        c, inv_2, sil_2, dur_2, coh_2, famp_2 = pulse_cost(uu, pulse, d; epoch_cost_kwargs..., compute=compute, solve_kwargs...)
        aux[] = (
            Float64(ForwardDiff.value(c)),
            Float64(ForwardDiff.value(inv_2)),
            Float64(ForwardDiff.value(sil_2)),
            Float64(ForwardDiff.value(dur_2)),
            Float64(ForwardDiff.value(coh_2)),
            Float64(ForwardDiff.value(famp_2)),
        )
        return c
    end
    grad = ForwardDiff.gradient(cost_only, u)
    cost, inv_, sil_, dur_, coh_, famp_ = aux[]
end

#Adam now perfectly optimizes the sandbox. However, the outer optimise_composite_pulse Basin Hopper must receive the strict, static cost to perform valid Metropolis tests between hops. At the very end of run_local_adam, intercept the return to evaluate the final state under the static baseline rules:
# Evaluate final true static cost for Basin Hopping
final_static_cost, _, _, _, _, _ = pulse_cost(best_u, pulse, d; cost_kwargs..., compute=compute, solve_kwargs...)
    
return best_u, final_static_cost, best_inv, best_sil, best_dur, history
end