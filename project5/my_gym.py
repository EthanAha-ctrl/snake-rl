import numpy as np

class Space:
    """
    Defines the observation and action spaces, so you can write generic
    code that applies to any Env. For example, you can choose a random
    action.
    """
    def __init__(self, shape=None, dtype=None):
        self.shape = shape
        self.dtype = dtype

    def sample(self):
        """
        Randomly sample an element of this space. Can be
        uniform or non-uniform sampling based on boundedness of space.
        """
        raise NotImplementedError

    def contains(self, x):
        """
        Return boolean specifying if x is a valid member of this space
        """
        raise NotImplementedError
    
    def __contains__(self, x):
        return self.contains(x)

class Discrete(Space):
    r"""
    A discrete space in :math:`\{ 0, 1, \\dots, n-1 \}`.
    """
    def __init__(self, n):
        self.n = n
        super().__init__((), np.int64)

    def sample(self):
        return np.random.randint(self.n)

    def contains(self, x):
        if isinstance(x, int):
            as_int = x
        elif isinstance(x, (np.generic, np.ndarray)) and (
            x.dtype.char in np.typecodes["AllInteger"] and x.shape == ()
        ):
            as_int = int(x)
        else:
            return False
        return 0 <= as_int < self.n

    def __repr__(self):
        return f"Discrete({self.n})"

class Box(Space):
    r"""
    A (possibly unbounded) box in :math:`\mathbb{R}^n`.
    Basically, an n-dimensional tensor.
    """
    def __init__(self, low, high, shape=None, dtype=np.float32):
        if shape is None:
            assert low.shape == high.shape, "box dimension mismatch"
            self.shape = low.shape
            self.low = low
            self.high = high
        else:
            assert np.isscalar(low) and np.isscalar(high)
            self.low = low + np.zeros(shape, dtype=dtype)
            self.high = high + np.zeros(shape, dtype=dtype)
            self.shape = shape
        self.dtype = dtype
        super().__init__(self.shape, self.dtype)

    def sample(self):
        return np.random.uniform(low=self.low, high=self.high, size=self.shape).astype(self.dtype)

    def contains(self, x):
        if not isinstance(x, np.ndarray):
            x = np.array(x, dtype=self.dtype)
        return x.shape == self.shape and np.all(x >= self.low) and np.all(x <= self.high)

    def __repr__(self):
        return f"Box({self.low}, {self.high}, {self.shape}, {self.dtype})"

class Env:
    """
    The main class. It encapsulates an environment with
    arbitrary behind-the-scenes dynamics. An environment can be
    partially or fully observed.
    """
    # Set this in SOME subclasses
    metadata = {'render.modes': []}
    reward_range = (-float('inf'), float('inf'))
    spec = None
    
    @property
    def unwrapped(self):
        """Completely unwrap this env."""
        return self

    def step(self, action):
        """Run one timestep of the environment's dynamics.
        Returns (obs, reward, terminated, truncated, info)
        """
        raise NotImplementedError

    def reset(self, seed=None, options=None):
        """Resets the state of the environment and returns an initial observation.
        Returns (obs, info)
        """
        raise NotImplementedError

    def render(self):
        """Renders the environment."""
        raise NotImplementedError

    def close(self):
        """Override close in your subclass to perform any necessary cleanup."""
        pass
