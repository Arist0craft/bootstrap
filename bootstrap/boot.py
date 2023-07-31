from multiprocessing import Pool
from typing import Callable, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def split(x, n):
    chunks = []
    if x < n:
        return -1

    elif x % n == 0:
        for i in range(n):
            chunks.append(x // n)

    else:
        zp = n - (x % n)
        pp = x // n
        for i in range(n):
            if i >= zp:
                chunks.append(pp + 1)
            else:
                chunks.append(pp)

    return chunks


class Bootstrap2D:
    def __init__(
        self,
        n_threads: int = 4,
        n_boot_samples: int = 10000,
        boot_sample_size: Union[int, str] = "MIN",
        replicate: Callable = np.mean,
    ):
        self._n_threads = n_threads
        self._n_boot_samples = n_boot_samples
        self._sample_size = boot_sample_size
        self._replicate = replicate
        self.replicates_diff_ = None

    def _get_sample_size(self, sample_a, sample_b):
        """
        Return sample size of bootstrap sample

        :param sample_a: first sample
        :param sample_b: second_sample
        :return: Return bootstrap sample size
        """
        if self._sample_size == "MAX":
            return max(len(sample_a), len(sample_b))
        elif self._sample_size == "MIN":
            return min(len(sample_a), len(sample_b))
        else:
            return self._sample_size

    def _calculate_replicate(self, sample_a, sample_b):
        """
        Calculate difference between replicates of sample_b and sample_a

        :param sample_a: first sample
        :param sample_b: second_sample
        """
        size = self._get_sample_size(sample_a, sample_b)

        boot_a_replicate = self._replicate(
            sample_a[np.random.choice(sample_a.shape[0], size, replace=True)]
        )
        boot_b_replicate = self._replicate(
            sample_b[np.random.choice(sample_b.shape[0], size, replace=True)]
        )

        return boot_b_replicate - boot_a_replicate

    def _iterate(self, sample_a, sample_b, iterations):
        """
        Iteratations in sub process

        :param sample_a: first sample
        :param sample_b: second_sample
        :return: Sub sample of replicates difference
        """
        replicates_diff = np.array(
            [self._calculate_replicate(sample_a, sample_b) for i in range(iterations)]
        )

        return replicates_diff

    def fit(self, sample_a, sample_b):
        """
        Calculate bootstrap samples replicates difference for confidence interval

        :param sample_a: first sample
        :param sample_b: second_sample
        """
        sample_a = np.array(sample_a)
        sample_b = np.array(sample_b)

        chunks = [
            (sample_a, sample_b, i)
            for i in split(self._n_boot_samples, self._n_threads)
        ]

        with Pool(self._n_threads) as pool:
            self.replicates_diff_ = np.concatenate(pool.starmap(self._iterate, chunks))

        return self

    def fit_transform(self, sample_a, sample_b):
        self.fit(sample_a, sample_b)
        return self.replicates_diff_

    def get_ci(self, confidence=0.95):
        alpha = (1 - confidence) * 100
        ci_start = np.percentile(self.replicates_diff_, alpha / 2)
        ci_end = np.percentile(self.replicates_diff_, 100 - alpha / 2)
        return (ci_start, ci_end)

    def plot(self, confidence=0.95):
        ci_start, ci_end = self.get_ci(confidence=confidence)
        _, ax = plt.subplots()
        sns.histplot(self.replicates_diff_, ax=ax)
        ax.axvline(ci_start, color="red")
        ax.axvline(ci_end, color="red")
        plt.show()
        print(f"Confidence interval: [{ci_start}, {ci_end}]")
