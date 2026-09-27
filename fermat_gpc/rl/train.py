"""CEM training loop with theoretical ceiling tracking."""
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from .policy import FermatPolicyNet
from ..enumeration import iter_sorted_tuples
from ..certificates import classify_tuple


def theoretical_ceiling(d: int) -> dict:
    """Compute (n_total, n_full, n_weak, R_max) at degree d."""
    n_total = n_full = n_weak = 0
    for tup in iter_sorted_tuples(d):
        n_total += 1
        cls = classify_tuple(tup, d)
        if cls["full_pair"]:
            n_full += 1
        if cls["weak"]:
            n_weak += 1
    R_max = n_full / n_total + 0.5 * (n_weak - n_full) / n_total
    return {"n_total": n_total, "n_full": n_full, "n_weak": n_weak,
            "R_max": R_max}


def train_cem(env, policy, epochs=15, batch_size=512, elite_frac=0.1,
              lr=1e-3, device="cpu", verbose=True):
    policy.to(device)
    optimiser = optim.Adam(policy.parameters(), lr=lr)
    history = {"mean_reward": [], "full_rate": [], "loss": []}
    for epoch in range(1, epochs + 1):
        tuples, at, ap, rw = [], [], [], []
        n_full = 0
        for _ in range(batch_size):
            obs = env.sample_state()
            xb = torch.tensor(obs, device=device).unsqueeze(0)
            with torch.no_grad():
                tl, pl = policy(xb)
                a_t = Categorical(logits=tl).sample().item()
                a_p = Categorical(logits=pl).sample().item()
            tup = tuple(int(round(v * env.d)) for v in obs)
            r, info = env.evaluate(tup, a_t, a_p)
            tuples.append(obs)
            at.append(a_t)
            ap.append(a_p)
            rw.append(r)
            n_full += int(info["full_certificate"])
        rw = np.array(rw, dtype=np.float32)
        k_elite = max(1, int(elite_frac * batch_size))
        elite_idx = np.argsort(-rw)[:k_elite]
        obs_e = torch.tensor(np.stack([tuples[i] for i in elite_idx]), device=device)
        at_e = torch.tensor([at[i] for i in elite_idx], dtype=torch.long, device=device)
        ap_e = torch.tensor([ap[i] for i in elite_idx], dtype=torch.long, device=device)
        optimiser.zero_grad()
        tl, pl = policy(obs_e)
        loss = F.cross_entropy(tl, at_e) + F.cross_entropy(pl, ap_e)
        loss.backward()
        optimiser.step()
        history["mean_reward"].append(float(rw.mean()))
        history["full_rate"].append(100.0 * n_full / batch_size)
        history["loss"].append(float(loss.item()))
        if verbose:
            print(f"[{epoch:02d}/{epochs}] mean_r={rw.mean():.3f} "
                  f"full={100*n_full/batch_size:5.1f}% loss={loss.item():.4f}")
    return history