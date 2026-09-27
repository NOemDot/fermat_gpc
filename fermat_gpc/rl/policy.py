"""S_6-equivariant policy with attention over index tokens."""
import torch
import torch.nn as nn


class FermatPolicyNet(nn.Module):
    def __init__(self, num_units: int, num_pairings: int = 15,
                 hidden: int = 64, num_indices: int = 6):
        super().__init__()
        self.num_indices = num_indices

        # Scalar encoder
        self.phi = nn.Sequential(
            nn.Linear(1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        # Positional embedding
        self.pos_emb = nn.Embedding(num_indices, hidden)
        # Attention
        self.attn = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        # Twist head (invariant)
        self.twist_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_units),
        )
        # Pairing head (equivariant scoring of 15 matchings)
        self.pair_scorer = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.register_buffer("pairings", self._pairing_tensor(num_indices))

    @staticmethod
    def _pairing_tensor(n):
        items = list(range(n))
        out = []

        def rec(rem, acc):
            if not rem:
                out.append(tuple(sorted(tuple(sorted(p)) for p in acc)))
                return
            first = rem[0]
            for i in range(1, len(rem)):
                second = rem[i]
                rest = [x for x in rem if x != first and x != second]
                rec(rest, acc + [(first, second)])

        rec(items, [])
        seen, uniq = set(), []
        for p in out:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        return torch.tensor(uniq, dtype=torch.long)

    def forward(self, x):
        # x: (B, 6)
        B, K = x.shape
        h = self.phi(x.view(-1, 1)).view(B, K, -1)
        pos = self.pos_emb(torch.arange(K, device=x.device))
        h = h + pos.unsqueeze(0)
        h_attn, _ = self.attn(h, h, h)
        h = self.norm(h + h_attn)
        pooled = h.mean(dim=1)
        twist_logits = self.twist_head(pooled)
        P = self.pairings
        idx1 = P[:, :, 0].reshape(-1)
        idx2 = P[:, :, 1].reshape(-1)
        h_i = h[:, idx1, :]
        h_j = h[:, idx2, :]
        pair_in = torch.cat([h_i, h_j], dim=-1)
        pair_scores = self.pair_scorer(pair_in).squeeze(-1).view(B, P.shape[0], P.shape[1])
        pairing_logits = pair_scores.sum(dim=-1)
        return twist_logits, pairing_logits