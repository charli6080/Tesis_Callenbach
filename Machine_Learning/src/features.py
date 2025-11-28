import numpy as np
import pandas as pd
from scipy.signal import find_peaks

def create_windows(df, window_size, step_size):
    x_list, y_list, z_list, gx_list, gy_list, gz_list, labels = [], [], [], [], [], [], []
    for i in range(0, df.shape[0] - window_size, step_size):
        xs = df['ax'].values[i: i + window_size]
        ys = df['ay'].values[i: i + window_size]
        zs = df['az'].values[i: i + window_size]
        gx = df['gx'].values[i: i + window_size]
        gy = df['gy'].values[i: i + window_size]
        gz = df['gz'].values[i: i + window_size]
        label = df['Actividades'][i: i + window_size].mode()[0]
        x_list.append(xs)
        y_list.append(ys)
        z_list.append(zs)
        gx_list.append(gx)
        gy_list.append(gy)
        gz_list.append(gz)
        labels.append(label)
    return x_list, y_list, z_list, gx_list, gy_list, gz_list, labels

def compute_features(x_df, y_df, z_df, gx_df, gy_df, gz_df, window_size):
    X = pd.DataFrame()
    acc_magnitude = np.sqrt(x_df**2 + y_df**2 + z_df**2)
    gyro_magnitude = np.sqrt(gx_df**2 + gy_df**2 + gz_df**2)
    for name, df in zip(
        ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'acc_mag', 'gyro_mag'],
        [x_df, y_df, z_df, gx_df, gy_df, gz_df, acc_magnitude, gyro_magnitude]
    ):
        X[f'{name}_mean'] = df.mean(axis=1)
        X[f'{name}_std'] = df.std(axis=1)
        X[f'{name}_median'] = df.median(axis=1)
        X[f'{name}_mad'] = np.median(np.abs(df - np.median(df, axis=1)[:, None]), axis=1)
        X[f'{name}_iqr'] = df.apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25), axis=1)
        X[f'{name}_peak_count'] = df.apply(lambda x: len(find_peaks(x)[0]), axis=1)
        X[f'{name}_energy'] = np.sum(df**2, axis=1) / window_size
    X['sma'] = (np.sum(np.abs(x_df), axis=1) + np.sum(np.abs(y_df), axis=1) + np.sum(np.abs(z_df), axis=1)) / window_size
    X['sma_gyro'] = (np.sum(np.abs(gx_df), axis=1) + np.sum(np.abs(gy_df), axis=1) + np.sum(np.abs(gz_df), axis=1)) / window_size

    # FFT Features
    def fft_basic_features(df):
        fft_vals = np.abs(np.fft.rfft(df, axis=1))
        return {
            'fft_mean': fft_vals.mean(axis=1),
            'fft_std': fft_vals.std(axis=1),
            'fft_max': fft_vals.max(axis=1),
            'fft_energy': np.sum(fft_vals**2, axis=1)
        }
    for name, df in zip(['ax', 'ay', 'az', 'gx', 'gy', 'gz'], [x_df, y_df, z_df, gx_df, gy_df, gz_df]):
        fft_feats = fft_basic_features(df)
        for k, v in fft_feats.items():
            X[f'{name}_{k}'] = v

    # Entropy
    def shannon_entropy(signal):
        hist, _ = np.histogram(signal, bins=10, density=True)
        hist += 1e-12
        return -np.sum(hist * np.log2(hist))
    for name, df in zip(['ax', 'ay', 'az', 'gx', 'gy', 'gz'], [x_df, y_df, z_df, gx_df, gy_df, gz_df]):
        X[f'{name}_entropy'] = df.apply(shannon_entropy, axis=1)

    # Correlation features
    X['corr_xy'] = [np.corrcoef(x, y)[0,1] for x, y in zip(x_df.values, y_df.values)]
    X['corr_xz'] = [np.corrcoef(x, z)[0,1] for x, z in zip(x_df.values, z_df.values)]
    X['corr_yz'] = [np.corrcoef(y, z)[0,1] for y, z in zip(y_df.values, z_df.values)]

    # Zero crossing rate
    def zero_crossings(signal):
        return ((signal[:, :-1] * signal[:, 1:]) < 0).sum(axis=1)
    X['x_zero_cross'] = zero_crossings(x_df.values)
    X['y_zero_cross'] = zero_crossings(y_df.values)
    X['z_zero_cross'] = zero_crossings(z_df.values)

    X.replace([np.inf, -np.inf], 0, inplace=True)
    X.fillna(0, inplace=True)
    return X
