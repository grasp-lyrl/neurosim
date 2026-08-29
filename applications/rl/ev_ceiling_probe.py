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
        for i in range(num_envs):
            open_traces[i].append(
                {
                    "state": np.asarray(raw["state"][i], dtype=np.float64),
                    "priv": np.asarray(raw["privileged"][i], dtype=np.float64),
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


def mlp_ev(Xtr, ytr, Xte, yte, epochs=400, hidden=256, seed=0):
    """A deliberately over-powered fit -- we want the CEILING, not a
    well-regularised model. If this cannot explain the return, nothing can."""
    th.manual_seed(seed)
    dev = "cuda:1" if th.cuda.is_available() else "cpu"
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    ym, ys = ytr.mean(), ytr.std() + 1e-8
    xtr = th.tensor((Xtr - mu) / sd, dtype=th.float32, device=dev)
    xte = th.tensor((Xte - mu) / sd, dtype=th.float32, device=dev)
    yt = th.tensor((ytr - ym) / ys, dtype=th.float32, device=dev).unsqueeze(1)
    net = th.nn.Sequential(
        th.nn.Linear(xtr.shape[1], hidden), th.nn.ReLU(),
        th.nn.Linear(hidden, hidden), th.nn.ReLU(),
        th.nn.Linear(hidden, 1),
    ).to(dev)
    opt = th.optim.Adam(net.parameters(), lr=1e-3)
    n = len(xtr)
    for _ in range(epochs):
        perm = th.randperm(n, device=dev)
        for i in range(0, n, 1024):
            idx = perm[i : i + 1024]
            loss = th.nn.functional.mse_loss(net(xtr[idx]), yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    with th.no_grad():
        pred = net(xte).cpu().numpy().reshape(-1) * ys + ym
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
    for name, X in [
        ("privileged(9) ridge", P),
        ("privileged(9) MLP", P),
        ("state+priv ridge", np.hstack([S, P])),
        ("state+priv MLP", np.hstack([S, P])),
        ("state+priv+HINDSIGHT MLP", np.hstack([S, P, H])),
    ]:
        fn = ridge_ev if "ridge" in name else mlp_ev
        print(f"{name:<34} {fn(X[tr], g_tr, X[te], g_te):>9.3f}", flush=True)


if __name__ == "__main__":
    main()
