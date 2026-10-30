# Key Sampling Engine

The workload generator samples keys for queries and writes over a sorted unique `uint64` dataset.

Selecting existing keys and sampling new keys in-domain operate in rank space, while kNN queries and boundary inserts operate in key space.
Rank space operations work on a symmetric window centered at the targeted percentile rank.
The rank window radius is the configurable `rank_spread` parameter (fraction of the dataset size, default `0.025`).

---

## Selecting Existing Records (Deletes and Queries)

- **Deletes:**
  - **Uniform:** Samples distinct indices uniformly at random without replacement.
  - **Gaussian, Zipfian:** Weighted selection without replacement using the [A-Res](https://en.wikipedia.org/wiki/Reservoir_sampling#Algorithm_A-Res) algorithm, where item weights decay with distance from the pivot.
- **Point and Range Queries:**
  - Draws existing keys *with replacement* to model hotspot access patterns.
  - If no percentile is specified, queries sample uniformly over the entire dataset.

---

## Sampling New Keys (Inserts)

To avoid rejection sampling pitfalls in dense key spaces, in-domain inserts use a **two-stage capacity-bounded allocation**:

1. **Gap Quota Allocation:**
   - Each gap between consecutive records $g_j = (k_{j-1}, k_j)$ has capacity $c_j = k_j - k_{j-1} - 1$ free integers.
   - Gaps with $c_j \ge 1$ receive quotas based on multinomial draws weighted by the chosen distribution kernel.
   - If a gap reaches capacity, it saturates and excess quota spills over to remaining non-full gaps in the window weighted by the distribution kernel.
2. **Key Placement:**
   - The allocated count of unique keys is drawn within each selected gap.
   - To prevent keys from scattering across huge empty voids, the drawable span is bounded by a multiple of the window's median gap scale.

### Boundary Inserts (Prepend / Append)
- **Prepend (`percentile <= 0`):** Generates strictly decreasing keys below $k_{\min}$ using cumulative positive steps scaled to the local gap density.
- **Append (`percentile >= 100`):** Generates strictly increasing keys above $k_{\max}$.

---

## Two-Phase Updates

Updates are emitted as two paired batches:
1. **Drain Batch (`delete`):** Drains existing keys selected around the target percentile.
2. **Refill Batch (`insert`):** Inserts new keys around the same percentile window.

Decomposing updates into two distinct batches avoids duplicate key collisions and allows refill keys to reuse slots vacated by the drain batch.

---

## kNN Query Points

- Samples continuous domain coordinates around the pivot key; drawn offsets are rounded to `uint64` integers and clamped to $[k_{\min}, k_{\max}]$.
- Both existing and non-existing keys are valid values for nearest-neighbor queries.
