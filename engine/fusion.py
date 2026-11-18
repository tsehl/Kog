"""Exact assignment with at most one replacement by a substitute. No card x card matrix.

Hall's condition on the subsets of the FK slots. After a substitute is reserved, the tight sets
(|neighborhood| = |set|) forbid certain profiles for the reserved substitute. A histogram of the 2**FK profiles
allows counting them in O(N).
"""
import jax
import jax.numpy as jnp


def completions(cand, substitutes, field, eligible, excluded):
    """Four tables [2**FK, M]: without/with substitute, without/with a material from the field.
    `excluded[M]`: candidate already reserved, -1 for none. The cards already chosen
    must have been removed from cand and substitutes by the caller.
    Cost O(FK^2 * 2**FK * M + FK * (N + FK * 2**FK)), memory O(FK^2 * 2**FK * M).
    FK=4 is a static bound; M=N for the choices, M=1 for availability.
    """
    fk, n = cand.shape
    q = jnp.arange(1 << fk, dtype=jnp.int32)
    slot = jnp.arange(fk)
    membership = (q[:, None] & (1 << slot)) != 0
    sizes = membership.sum(1)
    present = excluded >= 0
    ix = jnp.maximum(excluded, 0)
    allowed = (eligible & (1 << slot)) != 0

    def forced(l):
        candidates = cand & ((slot[:, None] != l) | field)
        bits = (candidates * (1 << slot[:, None])).sum(0)
        nq = 1 << fk
        # Histograms + zeta transform: no [Q,Q,N] tensor, nor [N,N].
        weights = jnp.stack([jnp.ones(n, jnp.int32), substitutes.astype(jnp.int32),
                             (substitutes & field).astype(jnp.int32)])
        hist = jnp.zeros((3,nq),jnp.int32).at[:,bits].add(weights)
        for j in range(fk):
            hist = hist + jnp.where((q & (1 << j))[None, :] != 0, hist[:, q ^ (1 << j)], 0)
        neighbors_excluded = (bits[ix][None, :] & q[:,None]) != 0
        counts = (hist[0,-1] - hist[0,(~q) & (nq-1)])[:,None] - (neighbors_excluded & present)
        bad = counts < sizes[:,None]
        forbidden = jnp.where(counts == sizes[:,None], q[:,None], 0)
        for j in range(fk):
            use = (q & (1 << j))[:,None] != 0
            bad = bad | (use & bad[q ^ (1 << j)])
            forbidden = forbidden | jnp.where(use, forbidden[q ^ (1 << j)], 0)
        plain = ~bad

        def wildcard(j):
            rest = q & ~(1 << j)
            ok = plain[rest]
            blocked = forbidden[rest]
            stock = jnp.where(l == j, hist[2], hist[1])[(~blocked) & (nq-1)]
            taken = substitutes[ix] & present & ((bits[ix][None, :] & blocked) == 0) & ((l != j) | field[ix])
            return ok & (stock - taken > 0) & allowed[j] & ((q & (1 << j)) != 0)[:, None]
        enhanced = plain | jax.vmap(wildcard)(slot).any(0)
        return plain, enhanced

    plain, enhanced = forced(jnp.int32(-1))
    fp, fs = jax.vmap(forced)(slot)
    use_field = membership.T[:, :, None]
    return plain, enhanced, (fp & use_field).any(0), (fs & use_field).any(0)
