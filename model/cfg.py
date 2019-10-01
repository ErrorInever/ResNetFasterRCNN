from easydict import EasyDict as edict

__C = edict()

# Consumers can get config by:
cfg = __C

# RPN
__C.ANCHOR = edict()
__C.ANCHOR.SCALES = [8, 16, 32]
__C.ANCHOR.RATIOS = [0.5, 1, 2]

# Normalization
__C.TRANSFORM = edict()
__C.TRANSFORM.MEAN = [0.485, 0.456, 0.406]
__C.TRANSFORM.STD = [0.229, 0.224, 0.225]

# Train
__C.TRAIN = edict()
__C.TRAIN.EPOCHS = 100
__C.TRAIN.LEARNING_RATE = 1.0e-3
__C.TRAIN.BATCH_SIZE = 128

# scheduler
__C.TRAIN.SCHEDULER = edict()
__C.TRAIN.SCHEDULER.STEP_SIZE = 5
__C.TRAIN.SCHEDULER.GAMMA = 0.1

# Stochastic gradient descent with momentum
__C.TRAIN.SGD = edict()
__C.TRAIN.SGD.MOMENTUM = 0.9

# L2 regularization
__C.WEIGHT_DECAY = 0.0005
