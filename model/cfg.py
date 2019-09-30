from easydict import EasyDict as edict

__C = edict()

# Consumers can get config by:
cfg = __C

# Anchor scales
__C.ANCHOR_SCALES = [8, 16, 32]

# Anchor ratios
__C.ANCHOR_RATIOS = [0.5, 1, 2]