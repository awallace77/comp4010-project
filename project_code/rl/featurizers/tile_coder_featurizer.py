
import numpy as np

class NumPyIHT:
    """
        A simple hash table like Sutton's IHT.
        Stores up to max_size entries. Collisions reuse existing indices.
    """
    def __init__(self, size):
        self.size = size
        self.dict = {}

    def __str__(self):
        return f"IHT(size={self.size}, used={len(self.dict)})"

    def get_index(self, obj):
        if obj in self.dict:
            return self.dict[obj]
        if len(self.dict) >= self.size:
            # Hash collision: use modular hash
            return hash(obj) % self.size
        index = len(self.dict)
        self.dict[obj] = index
        return index


def tiles(iht, num_tilings, floats):
    """
        Tile coding
        Returns a list of active tile indices (one per tiling)
    """

    dims = len(floats)

    # Offsets for each tiling (gives displacement)
    offsets = (np.arange(num_tilings)[:, None] * 2.0) / num_tilings

    # Shifted inputs for each tiling
    shifted = (floats + offsets) * num_tilings

    # Floor to get tile coordinates
    coords = np.floor(shifted).astype(int)

    # Append tiling index (ensures each tiling maps to separate region)
    tiling_ids = np.arange(num_tilings)[:, None]
    coords = np.concatenate([coords, tiling_ids], axis=1)

    # Hash each coordinate tuple into an integer feature
    active = []
    for c in coords:
        key = tuple(c.tolist())
        idx = iht.get_index(key)
        active.append(idx)

    return active

class TileCoder:
    """
        Returns a dense feature vector of length max_size with num_tilings active bits.
    Args:
        env: gym environment
        num_tilings: number of overlapping tilings (16 is a good default)
        tiles_per_dim: scaling parameter (5-10 typical)
        max_size: number of hashed features (controls featurizer dimension)
    """
    def __init__(self, env, num_tilings=16, tiles_per_dim=8, max_size=4096):
        self.num_tilings = num_tilings
        self.tiles_per_dim = tiles_per_dim
        self.iht = NumPyIHT(max_size)
        self.n_features = max_size

        low = env.observation_space.low
        high = env.observation_space.high

        # Replace infinite bounds
        low = np.where(np.isfinite(low), low, -1.0)
        high = np.where(np.isfinite(high), high, 1.0)

        self.low = low
        self.high = high

    def _scale_state(self, state):
        """
            Normalize each dimension into [0, tiles_per_dim].
        """
        return (state - self.low) / (self.high - self.low + 1e-12) * self.tiles_per_dim

    def featurize(self, state):
        """
            Returns a dense feature vector with num_tilings active tiles
        """
        scaled = self._scale_state(state)

        # Get active tile indices
        active_tiles = tiles(self.iht, self.num_tilings, scaled.tolist())

        # Convert to dense binary feature vector
        x = np.zeros(self.n_features)
        x[active_tiles] = 1.0
        return x

