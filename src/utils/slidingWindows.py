import numpy as np

# determine sliding window (period) based on ACF
def _acf_fft(data: np.ndarray, nlags: int) -> np.ndarray:
    data = np.asarray(data, dtype=float)
    data = data - np.mean(data)
    n = data.size
    if n == 0:
        return np.zeros(nlags + 1, dtype=float)

    # FFT-based autocorrelation (O(n log n)).
    fft = np.fft.rfft(data, n=2 * n)
    acf = np.fft.irfft(fft * np.conjugate(fft))[: nlags + 1]
    if acf[0] == 0:
        return acf
    return acf / acf[0]


def _local_maxima(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.size < 3:
        return np.array([], dtype=int)
    return np.where((values[1:-1] > values[:-2]) & (values[1:-1] > values[2:]))[0] + 1


def find_length(data):
    if len(data.shape)>1:
        return 0
    data = data[:min(20000, len(data))]
    
    base = 3
    auto_corr = _acf_fft(data, nlags=400)[base:]

    local_max = _local_maxima(auto_corr)
    if local_max.size == 0:
        return 125

    best = local_max[np.argmax(auto_corr[local_max])]
    if best < 3 or best > 300:
        return 125
    return int(best + base)
