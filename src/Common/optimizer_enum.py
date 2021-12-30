from enum import Enum


class Optimizer(Enum):
    AdaDelta    = 'Adadelta'
    AdaGrad     = 'Adagrad'
    Adam        = 'Adam'
    AdaMax      = 'Adamax'
    Ftrl        = 'Ftrl'
    NAdam       = 'Nadam'
    RMSprop     = 'RMSprop'
    SGD         = 'SGD'
