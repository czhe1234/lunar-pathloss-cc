from pathlib import Path
import json, math, hashlib
import numpy as np

ROOT = Path("/home/chenzhe/work/lunar-pathloss-rm-challenge/CC")
OUT = ROOT / "audit/learning_20261110"
OUT.mkdir(parents=True, exist_ok=True)
result = {"scope": "Analytic counterexamples only; no competition samples, no model training, no GPU.", "date": "2026-11-10"}

g = 1e-8
draws = np.array([0.5*g, 1.5*g], dtype=np.float64)
power_mean = float(draws.mean())
mean_pl = float((-10*np.log10(draws)).mean())
pl_of_mean = -10*math.log10(power_mean)
assert math.isclose(power_mean, g, rel_tol=1e-14)
assert mean_pl > pl_of_mean
result["linear_unbiased_not_db_unbiased"] = {
    "true_gain": g, "equally_likely_gains": draws.tolist(),
    "mean_gain": power_mean, "PL_of_mean_gain_db": pl_of_mean,
    "mean_PL_db": mean_pl, "bias_db": mean_pl-pl_of_mean,
    "assumptions": "Positive unmasked unclipped power. Selection by validity masks or clipping adds further changes."
}

k0 = 2*math.pi
angles = np.array([0., math.pi/3])
result["power_not_complex_field"] = {
    "construction": "E(x,z)=exp(i*k0*(sin(theta)*x+cos(theta)*z)), restricted to z=0.",
    "k0": k0, "theta_radians": angles.tolist(),
    "power_both": [1.,1.],
    "minus_surface_laplacian_over_E": (k0*k0*np.sin(angles)**2).tolist(),
    "minus_surface_laplacian_over_sqrt_power": [0.,0.],
    "meaning": "Equal scalar power can coexist with different phase gradients and surface Helmholtz ratios. Not a measurement of dataset ambiguity."
}
assert result["power_not_complex_field"]["minus_surface_laplacian_over_E"][1] > 0

K = np.array([[0.,0.1],[0.2,0.]])
Q = np.array([[1.,0.3],[0.4,1.2]]) # two TX columns
A = np.array([[0.7,0.3],[0.2,0.8],[0.4,0.6]])
T = np.diag([2.,0.5])
Ti = np.linalg.inv(T)
K2, Q2, A2 = T@K@Ti, T@Q, A@Ti
L = np.linalg.solve(np.eye(2)-K,Q)
L2 = np.linalg.solve(np.eye(2)-K2,Q2)
Y, Y2 = A@L, A2@L2
err = float(np.max(np.abs(Y-Y2)))
assert err < 1e-12
assert np.max(np.abs(L-L2)) > 0.1
result["multiple_TX_do_not_identify_free_internal_states"] = {
    "K": K.tolist(), "Q_two_TX": Q.tolist(), "A": A.tolist(), "state": L.tolist(),
    "K_transformed": K2.tolist(), "Q_transformed": Q2.tolist(),
    "A_transformed": A2.tolist(), "state_transformed": L2.tolist(),
    "max_observation_difference": err,
    "scope": "A,K,Q all free. Known physical A,Q, units and coupling can remove this freedom. Not a proof of unidentifiability for the competition."
}

residual = 1.1 - 0.2 - 0.8*1.1
bound = abs(residual)/(1.-0.8)
assert math.isclose(bound, 0.1, abs_tol=1e-12)
result["fixed_transport_residual_bound"] = {
    "K": 0.8, "Q": 0.2, "true_state": 1., "candidate_state": 1.1,
    "residual": residual, "error_bound": bound,
    "scope": "Known correct K,Q and operator norm strictly below one; it does not validate geometry or labels."
}
result["one_db_pointwise_gain_interval"] = [10**(-0.1),10**0.1]
a = np.array([0.1+0j,-0.1+0j])
result["coherent_vs_noncoherent"] = {"coherent_power":float(abs(a.sum())**2),"noncoherent_power":float((abs(a)**2).sum())}


k_draws = np.array([0.25,0.75],dtype=np.float64)
k_mean, k_var = float(k_draws.mean()), float(k_draws.var())
q0 = 1.
true_x = q0/(1-k_mean)
biased_x = q0*(1-k_mean)/((1-k_mean)**2+k_var)
def noisy_residual_square(x):
    return float(np.mean((x-q0-k_draws*x)**2))
assert math.isclose(true_x,2.)
assert math.isclose(biased_x,1.6)
assert noisy_residual_square(biased_x) < noisy_residual_square(true_x)
result["squared_MC_residual_bias"] = {
    "K_samples":k_draws.tolist(),"Q":q0,"mean_K":k_mean,"variance_K":k_var,
    "true_fixed_point":true_x,"minimum_of_expected_single_sample_squared_residual":biased_x,
    "objective_at_true":noisy_residual_square(true_x),
    "objective_at_biased":noisy_residual_square(biased_x),
    "PL_difference_db_if_state_is_gain":-10*math.log10(biased_x/true_x),
    "scope":"Constructed scalar example. A noisy unbiased operator estimate need not yield an unbiased squared residual objective. Not a finding about the released dataset or a reproduction of Neural Radiosity."
}

d = json.loads((ROOT/"audit/c138_state/summary.json").read_text())
d139 = json.loads((ROOT/"audit/c139_bottleneck/summary.json").read_text())
result["verified_local_evidence"] = {
    "C138_state_supervision_16_FIT_RMSE":d["C138_state_auxiliary"]["last"]["training"]["RMSE"],
    "C138_matched_continuation_16_FIT_RMSE":d["C138_continuation"]["last"]["training"]["RMSE"],
    "C138_limitations":d["limits"],
    "C139_fixed_D_model_class_optimistic_RMSE":d139["fixed_D_class_optimistic_monitor_RMSE"],
    "C139_local_remap_radius4_optimistic_RMSE":d139["local_C82_remap_optimistic_monitor"]["4"],
    "scope":"C138=16 fixed FIT maps, sampled points. C139=128 terrain monitor sampled points. Neither is official validation.",
    "source_sha256": {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in [
        "src/c136_model.py","audit/c138_state/summary.json","audit/c139_bottleneck/summary.json"]}
}
path = OUT/"assumption_checks.json"
path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
print(json.dumps({"checks":"passed","output":str(path),"db_bias":mean_pl-pl_of_mean,"multi_TX_same_output_maxdiff":err,"GPU_used":False},ensure_ascii=False))
