"""Command-line naming of models/basic.py: result-file names, variant tokens and the QIF / LIF override flags."""
from mapstdp import lif
from models.helpers.agent import GAMMA
from models.helpers.tasks import TASKS

LA_BETA, HEADS = 0.3, (1, 2, 4, 8, 16)


def name(task, backend, heads, la_beta, seed, td_norm=0.0, variant=""):
    return f"m3_{task}_{backend}" + (f"_h{heads}" if backend == "spiking" else "") + f"_lb{la_beta:g}" + \
        (f"_tn{td_norm:g}" if td_norm else "") + (f"_{variant}" if variant else "") + f"_s{seed}"


def variant_tokens(a):
    """File-name tokens of the variants (module doc), in file-name order; empty on the defaults."""
    t = ([f"lam{a.lam:g}"] if a.lam != 0.05 else []) + ([f"pcc{a.p_cc:g}"] if a.p_cc != 0.1 else [])
    t += [f"px{a.p_cross:g}"] if a.p_cross != 0.05 else []
    t += [f"b{a.beta:g}"] if a.beta != 1.0 else []
    t += [f"bk{a.beta_kappa:g}m{a.beta_max:g}"] if a.beta_kappa else []
    t += [f"g{a.gamma:g}"] if a.gamma != GAMMA else []
    t += ["dbc"] if a.readout == "diffbound" else []
    t += ([f"enc{a.encoder}"] if a.encoder else []) + ([f"crit{a.critic_layer}"] if a.critic_layer else [])
    t += [f"ce{a.c_ent:g}" + (f"H{a.ent_target:g}" if a.ent_target is not None else "")] if a.c_ent else []
    t += [f"eta{a.eta:g}"] if a.eta is not None else []
    t += [qif_token(a)] if a.engine == "qif" and a.backend == "spiking" else []
    t += [lif_token(a)] if a.engine == "lif" and a.backend == "spiking" else []
    t += [qif_token(a).replace("qif", "q", 1)] if a.backend == "mpr" and qif_overrides(a) else []
    if a.rule != "v4":
        t += ["-".join([a.rule] + ([f"ot{a.obs_tau:g}"] if a.obs_tau else [])
                       + (["ebmod"] if a.ebar == "module" else []) + (["stk"] if a.step_stat == "kappa" else []))]
    t += ([f"ls{a.lam_scale:g}"] if a.lam_scale != 1.0 else []) + ([f"es{a.eta_scale:g}"] if a.eta_scale != 1.0 else [])
    t += ["ms"] if a.match_scales else []
    return t + ([f"ep{a.episodes}"] if a.episodes and a.episodes != TASKS[a.task].episodes else [])


QIF_FLAGS = {"tau_s": float, "inh": str, "warm": int, "lags": int, "tau_plus": float, "gain": float, "rate": float,
             "a_ext": float, "delta": float, "eta_bar": float, "eta_max": float, "sigma": float, "k": int,
             "burn": int, "w_ei": float, "w_ie": float, "n_inh": int, "tau": float, "lam_scale": float,
             "eta_scale": float, "eps_h_scale": float}  # --engine qif: QIFParams fields (--q-<field>)


def qif_overrides(a):
    """QIFParams overrides given on the command line (--q-<field>), in QIF_FLAGS order."""
    return {k: getattr(a, f"q_{k}") for k in QIF_FLAGS if getattr(a, f"q_{k}") is not None}


def qif_token(a):
    """File-name token of --engine qif: qif, then -<field><value> per override."""
    o = qif_overrides(a)
    return "-".join(["qif", *(f"{k}{v if isinstance(v, str) else format(v, 'g')}" for k, v in o.items())])


LIF_FLAGS = {"tau_m": float, "dt": float, "tau_ref": float, "mu_bg": float, "sigma_V": float, "delta_V": float,
             "rho0": float, "reset": int, "chi": float, "a_ext": float, "ext_rate": float, "k": int, "window": float,
             "counter": str, "omega": float, "gain_tau": float, "reg_tau": float, "reg_min": int, "warm": int,
             "warmup": int, "lam_scale": float, "eta_scale": float,
             "eps_h_scale": float}  # --engine lif: LIFParams fields (--lif-<field>)


def lif_overrides(a):
    """LIFParams overrides given on the command line (--lif-<field>), in LIF_FLAGS order (int flags of bool fields
    become bools)."""
    o = {k: getattr(a, f"lif_{k}") for k in LIF_FLAGS if getattr(a, f"lif_{k}") is not None}
    return {k: bool(v) if isinstance(getattr(lif.LIFParams(), k), bool) else v for k, v in o.items()}


def lif_token(a):
    """File-name token of --engine lif: lif, then -<field><value> per override."""
    o = {k: getattr(a, f"lif_{k}") for k in LIF_FLAGS if getattr(a, f"lif_{k}") is not None}
    return "-".join(["lif", *(f"{k}{v if isinstance(v, str) else format(v, 'g')}" for k, v in o.items())])
