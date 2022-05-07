import matplotlib.pyplot as plt
import numpy as np
from typing import Iterable

from .utils import tonp
# from utils import tonp

class BackdoorParetoPlot:
    def __init__(self, clean_perf: Iterable[float], backdoor_perf: Iterable[float], maximise: bool=True) -> None:
        clean_perf = np.array(clean_perf)
        backdoor_perf = np.array(backdoor_perf)

        if maximise:
            # Higher perf numbers are better (e.g. accuracy)

            # We assume the best perf is without any backdoor
            best_perf = np.max(clean_perf)

            backdoor_argsort = np.argsort(-backdoor_perf)

            backdoor_perf = backdoor_perf[backdoor_argsort]
            clean_perf = clean_perf[backdoor_argsort]

            pareto_backdoor_perf = []
            pareto_clean_perf = []
            for b, c in zip(backdoor_perf, clean_perf):
                if (not pareto_clean_perf) or c > pareto_clean_perf[-1]:
                    pareto_backdoor_perf.append(b)
                    pareto_clean_perf.append(c)

            # print(pareto_backdoor_perf)
            # print(pareto_clean_perf)
            
            plt.plot(pareto_backdoor_perf, pareto_clean_perf, color='red')
            plt.scatter(backdoor_perf, clean_perf, alpha=0.25)
            plt.plot((0, 1), (best_perf, best_perf), '--', color='green', alpha=0.5)

            # plt.xlim(min(pareto_backdoor_perf)-0.01, max(pareto_backdoor_perf))
            plt.xlim(0, 1)
            plt.ylim(0.8, max(pareto_clean_perf)+0.01)

            plt.xlabel('Backdoor Success Rate')
            plt.ylabel('Clean Accuracy')

        else:
            raise NotImplementedError
