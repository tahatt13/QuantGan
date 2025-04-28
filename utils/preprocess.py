import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import wasserstein_distance


def rolling_window(x, k, sparse=True):
    """compute rolling windows from timeseries

    Args:
        x ([2d array]): x contains the time series in the shape (timestep, sample).
        k ([int]): window length.
        sparse (bool): Cut off the final windows containing NA. Defaults to True.

    Returns:
        [3d array]: array of rolling windows in the shape (window, timestep, sample).
    """
    out = np.full([k, *x.shape], np.nan)
    N = len(x)
    for i in range(k):
        out[i, :N-i] = x[i:]

    if not sparse:
        return out

    return out[:, :-(k-1)]

def acf(x, k, le=False):

    arr = rolling_window(x, k, sparse=False)
    a = (arr[0] - np.nanmean(arr[0], axis=0))
    if le:
        arr **=2
    b = (arr - np.nanmean(arr, axis=1, keepdims=True))

    return np.nansum((a * b), axis=1) / np.sqrt(np.nansum(a**2, axis=0) * np.nansum(b**2, axis=1))

def compute_metrics(real_returns, fake_samples, windows=[1,5,20,100], acf_lags=1):
    """
    Compute EMD, DY and Leverage Effect between real and generated samples
    Args:
        real_returns (np.array): (T,) array of real returns
        fake_samples (np.array): (N_simulations, T) array of synthetic returns
    Returns:
        metrics (dict): all computed metrics
    """
    metrics = {}

    # EMD and DY
    for k in windows:
        real_cum = rolling_window(real_returns, k, sparse=not (k==1)).sum(axis=0).ravel()
        fake_cum = rolling_window(fake_samples.T, k, sparse=not (k==1)).sum(axis=0).ravel()

        emd = wasserstein_distance(real_cum, fake_cum)
        metrics[f"EMD({k})"] = emd

        vol_real = np.std(real_cum) * np.sqrt(252 / k)
        vol_fake = np.std(fake_cum) * np.sqrt(252 / k)
        dy = abs(vol_real - vol_fake)
        metrics[f"DY({k})"] = dy

    # Leverage effect
    lev_real = acf(real_returns, 100, le=True)[acf_lags-1]
    lev_fake = acf(fake_samples.T, 100, le=True).mean(axis=1)[acf_lags-1]
    metrics["Leverage Effect"] = np.abs(lev_real - lev_fake)

    return metrics