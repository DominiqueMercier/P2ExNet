import os

import numpy as np


def compute_prototype_distances(proto_qualities):
    """Prints the prototype quality. Therefore the distances are used.

    Args:
        proto_qualities (array): Array with the distances of the prototypes.
    """
    proto_best = int(np.min(proto_qualities) * 10000) / 10000
    proto_worst = int(np.max(proto_qualities) * 10000) / 10000
    proto_avg = int(np.average(proto_qualities) * 10000) / 10000
    proto_variance = int(
        (np.max(proto_qualities) - np.min(proto_qualities)) * 10000) / 10000
    proto_std = int(np.std(proto_qualities) * 10000) / 10000
    print('#'*80)
    print('Protoype Quality:')
    print('Min: %s | Max: %s | Avg: %s | Var: %s | Std: %s' %
          (proto_best, proto_worst, proto_avg, proto_variance, proto_std))
    print('%s & %s & %s & %s & %s' %
          (proto_best, proto_worst, proto_avg, proto_variance, proto_std))
