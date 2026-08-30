"""How much of the return is explainable AT ALL? -- the EV ceiling.

Two PPO runs (v16_effnet, v17_effnet, ~915k steps combined) sat at
explained_variance ~ 0 while ep_rew_mean stayed flat near -165, well short
of a blind 0.5 Hz sine weave at -122.1. Three hypotheses have now been
tried and falsified at the hyperparameter level (memory, rollout horizon,
encoder capacity). This asks a different question: is EV ~ 0 a broken
critic, or the task's ceiling?

The critic's input is deliberately narrow. velocity_dodge.py's
make_privileged_observation returns PRIVILEGED_DIM=9 -- a presence flag,
body-frame rel_pos(3), rel_vel(3), clearance, and time-to-closest-approach
-- for the NEAREST obstacle only, and its docstring says the omission of
the spawn schedule and future throws is deliberate. If return variance is
dominated by which obstacle gets thrown next and when, then no critic
reading those 9 numbers can explain it, EV ~ 0 is CORRECT behaviour, and
every further hyperparameter run is wasted.

Method: roll out the trained policy, compute discounted return-to-go, and
fit predictors of increasing information, holding out whole episodes.

    constant          EV = 0 by construction (the reference point)
    privileged(9)     exactly what the critic sees
    state+privileged  everything any network in this setup could use
    + HINDSIGHT       steps_remaining and did-it-crash: future facts NO
                      causal predictor can have. This is the upper bound
                      on how much of the return is explainable at all.

Reading it: if state+privileged tops out near the ~0.05 PPO reached, the
critic is already at its ceiling and the fix is in the task/reward, not
the network. If HINDSIGHT is also low, the return is mostly irreducible
noise. If HINDSIGHT is high while state+privileged is low, the gap is
precisely the future information the critic is denied.
"""
import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from train_sb3 import load_experiment_config, make_env


def collect(model, vec_env, n_episodes, gamma, num_envs):
    """Roll out until n_episodes have FINISHED, keeping per-episode traces."""
    open_traces = [[] for _ in range(num_envs)]
    done_eps = []
    obs = vec_env.reset()
    while len(done_eps) < n_episodes:
        with th.no_grad():
            obs_t, _ = model.policy.obs_to_tensor(obs)
            values = model.policy.predict_values(obs_t).cpu().numpy().reshape(-1)
        action, _ = model.predict(obs, deterministic=False)
        raw = vec_env.get_original_obs()
        # Dict observations carry "state" and a critic-only "privileged"
        # channel. A privileged-ACTOR config has neither: obs_mode: state
        # with privileged_critic: false gives a plain Box whose state vector
        # ALREADY contains the obstacle features. Handle both, so this probe
        # runs against the privileged teacher as well as the event configs.
        if isinstance(raw, dict):
            states_i, privs_i = raw["state"], raw.get("privileged")
        else:
            states_i, privs_i = raw, None
        for i in range(num_envs):
            open_traces[i].append(
                {
                    "state": np.asarray(states_i[i], dtype=np.float64),
                    "priv": (
                        np.zeros(0, dtype=np.float64)
                        if privs_i is None
                        else np.asarray(privs_i[i], dtype=np.float64)
                    ),
                    "value": float(values[i]),
                }
            )
        obs, reward, done, infos = vec_env.step(action)
        for i in range(num_envs):
            open_traces[i][-1]["reward"] = float(reward[i])
        for i in range(num_envs):
            if not done[i]:
                continue
            # SB3 VecEnv auto-resets; "TimeLimit.truncated" marks a cut
            # episode, whose return-to-go would need a bootstrap. Keep only
            # naturally-terminated episodes so return-to-go is exact.
            truncated = bool(infos[i].get("TimeLimit.truncated", False))
            trace = open_traces[i]
            open_traces[i] = []
            if truncated or len(trace) < 5:
                continue
            done_eps.append(trace)
            if len(done_eps) % 20 == 0:
                print(f"  collected {len(done_eps)} episodes", flush=True)
    return done_eps[:n_episodes]


def build_matrices(episodes, gamma):
    """Per-step features and discounted return-to-go, plus episode ids."""
    S, P, G, V, H, E = [], [], [], [], [], []
    for ep_id, trace in enumerate(episodes):
        n = len(trace)
        rewards = np.array([t["reward"] for t in trace], dtype=np.float64)
        rtg = np.zeros(n)
        acc = 0.0
        for k in range(n - 1, -1, -1):
            acc = rewards[k] + gamma * acc
            rtg[k] = acc
        # Hindsight: facts about the FUTURE that no causal predictor can
        # have. steps_remaining and the terminal outcome between them pin
        # down when and how the episode ends.
        crashed = float(rewards[-1] < -5.0)
        for k, t in enumerate(trace):
            S.append(t["state"])
            P.append(t["priv"])
            V.append(t["value"])
            G.append(rtg[k])
            H.append([n - k, crashed, (n - k) * crashed])
            E.append(ep_id)
    return (
        np.asarray(S), np.asarray(P), np.asarray(G),
        np.asarray(V), np.asarray(H, dtype=np.float64), np.asarray(E),
    )


def ev(y, pred):
    return 1.0 - np.var(y - pred) / np.var(y)


def ridge_ev(Xtr, ytr, Xte, yte, lam=1.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    A = np.hstack([(Xtr - mu) / sd, np.ones((len(Xtr), 1))])
    B = np.hstack([(Xte - mu) / sd, np.ones((len(Xte), 1))])
    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ ytr)
    return ev(yte, B @ w)


def mlp_ev(Xtr, ytr, Xte, yte, groups_tr=None, epochs=400, hidden=256, seed=0):
    """Nonlinear ceiling estimate, regularised and early-stopped.

    The original version here was "deliberately over-powered -- we want the
    CEILING, not a well-regularised model", trained 400 epochs with no weight
    decay, no validation split and no early stopping. On ~14k rows with a
    256-wide net that does not measure a ceiling, it measures overfitting:
    it returned EV -1.26 and -0.67, i.e. WORSE THAN A CONSTANT, which is
    impossible for a genuine ceiling (a constant is always achievable). Those
    numbers were correctly flagged as unusable, leaving only the ridge rows.

    An unregularised fit cannot bound what is explainable, because the
    quantity it maximises is training fit, and test EV falls as it succeeds.
    So: weight decay, and early stopping on a validation split held out BY
    EPISODE. Grouping matters -- consecutive steps in one episode share
    nearly all of their return, so a row-level split leaks the answer and
    would report an optimistic ceiling.

    ``groups_tr`` is the episode id per training row. Without it the split
    falls back to row-level and the result is reported as untrustworthy by
    the caller.
    """
    th.manual_seed(seed)
    dev = "cuda:0" if th.cuda.is_available() else "cpu"
    rng = np.random.default_rng(seed)

    # Validation split, held out by episode.
    if groups_tr is not None:
        gids = np.unique(groups_tr)
        rng.shuffle(gids)
        va_ids = set(gids[: max(1, len(gids) // 5)].tolist())
        va_mask = np.array([g in va_ids for g in groups_tr])
    else:
        va_mask = rng.random(len(Xtr)) < 0.2
    if va_mask.all() or not va_mask.any():
        va_mask = rng.random(len(Xtr)) < 0.2

    Xf, yf = Xtr[~va_mask], ytr[~va_mask]
    Xv, yv = Xtr[va_mask], ytr[va_mask]

    mu, sd = Xf.mean(0), Xf.std(0) + 1e-8
    ym, ys = yf.mean(), yf.std() + 1e-8
    xf = th.tensor((Xf - mu) / sd, dtype=th.float32, device=dev)
    xv = th.tensor((Xv - mu) / sd, dtype=th.float32, device=dev)
    xte = th.tensor((Xte - mu) / sd, dtype=th.float32, device=dev)
    yt = th.tensor((yf - ym) / ys, dtype=th.float32, device=dev).unsqueeze(1)
    yvt = th.tensor((yv - ym) / ys, dtype=th.float32, device=dev).unsqueeze(1)

    net = th.nn.Sequential(
        th.nn.Linear(xf.shape[1], hidden), th.nn.ReLU(),
        th.nn.Dropout(0.1),
        th.nn.Linear(hidden, hidden), th.nn.ReLU(),
        th.nn.Dropout(0.1),
        th.nn.Linear(hidden, 1),
    ).to(dev)
    opt = th.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)

    n = len(xf)
    best_val, best_state, patience, since = float("inf"), None, 25, 0
    for _ in range(epochs):
        net.train()
        perm = th.randperm(n, device=dev)
        for i in range(0, n, 1024):
            idx = perm[i : i + 1024]
            loss = th.nn.functional.mse_loss(net(xf[idx]), yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with th.no_grad():
            v = float(th.nn.functional.mse_loss(net(xv), yvt))
        if v < best_val - 1e-5:
            best_val, since = v, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            since += 1
            if since >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with th.no_grad():
        pred = net(xte).cpu().numpy().reshape(-1) * ys + ym
    # A ceiling can never be worse than a constant: report max(EV, 0) would
    # hide a bug, so return the raw value and let a negative one stand as the
    # signal that even a regularised fit found nothing.
    return ev(yte, pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="applications/rl/configs/velocity_dodge_dynamic_v17_effnet.yaml")
    ap.add_argument("--checkpoint", default="outputs/rl/v17_effnet_ppo/checkpoints/policy_589824_steps.zip")
    ap.add_argument("--vecnormalize", default="outputs/rl/v17_effnet_ppo/vecnormalize.pkl")
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--gpu-envs", type=int, default=1)
    args = ap.parse_args()

    exp = load_experiment_config(args.config)
    gamma = float(exp["ppo"]["gamma"])
    num_envs = int(exp["num_envs"])
    env_config = copy.deepcopy(exp["env"])
    env_config["enable_visualization"] = False

    vec = SubprocVecEnv([
        make_env(env_config, seed=12345 + i, env_idx=i, train=True, gpu_id=args.gpu_envs)
        for i in range(num_envs)
    ])
    vec = VecNormalize.load(args.vecnormalize, vec)
    vec.training = False
    vec.norm_reward = True

    model = PPO.load(args.checkpoint, env=vec, device="cuda:1")
    print(f"loaded {args.checkpoint}; gamma={gamma} num_envs={num_envs}", flush=True)

    episodes = collect(model, vec, args.episodes, gamma, num_envs)
    vec.close()
    S, P, G, V, H, E = build_matrices(episodes, gamma)
    print(f"\n{len(episodes)} episodes, {len(G)} steps; "
          f"state dim={S.shape[1]} priv dim={P.shape[1]}", flush=True)

    # Hold out whole EPISODES -- splitting by step would leak, since
    # consecutive steps in one episode share nearly all of their return.
    ids = np.unique(E)
    rng = np.random.default_rng(0)
    rng.shuffle(ids)
    te_ids = set(ids[: max(1, len(ids) // 5)].tolist())
    te = np.array([e in te_ids for e in E])
    tr = ~te
    print(f"train {tr.sum()} steps / test {te.sum()} steps "
          f"({len(ids) - len(te_ids)}/{len(te_ids)} episodes)\n", flush=True)

    g_tr, g_te = G[tr], G[te]
    between = np.var([G[E == e].mean() for e in ids]) / np.var(G)
    print(f"return-to-go: mean={G.mean():.2f} sd={G.std():.2f}")
    print(f"between-episode share of variance: {between:.1%}\n")

    print(f"{'predictor':<34} {'EV(test)':>9}")
    print("-" * 45)
    print(f"{'constant (reference)':<34} {0.0:>9.3f}")
    print(f"{'PPO critic (as trained)':<34} {ev(g_te, V[te]):>9.3f}")
    # With a privileged ACTOR the obstacle features live inside S and P is
    # empty, so the "privileged only" rows are meaningless; skip them rather
    # than print a fit on a zero-width matrix.
    rows = []
    if P.shape[1] > 0:
        rows += [
            (f"privileged({P.shape[1]}) ridge", P),
            (f"privileged({P.shape[1]}) MLP", P),
        ]
    rows += [
        (f"state+priv({S.shape[1] + P.shape[1]}) ridge", np.hstack([S, P])),
        (f"state+priv({S.shape[1] + P.shape[1]}) MLP", np.hstack([S, P])),
        ("state+priv+HINDSIGHT MLP", np.hstack([S, P, H])),
    ]
    for name, X in rows:
        if "ridge" in name:
            val = ridge_ev(X[tr], g_tr, X[te], g_te)
        else:
            val = mlp_ev(X[tr], g_tr, X[te], g_te, groups_tr=E[tr])
        print(f"{name:<34} {val:>9.3f}", flush=True)

    print(
        "\nMLP rows are now early-stopped on an episode-grouped validation\n"
        "split with weight decay. The previous unregularised version returned\n"
        "-1.26 and -0.67 -- worse than a constant, which is overfitting, not a\n"
        "ceiling. Compare the ridge and MLP rows: if they agree, the return is\n"
        "as linear as it is predictable, and the gap to HINDSIGHT is the part\n"
        "no causal predictor can reach."
    )


if __name__ == "__main__":
    main()
