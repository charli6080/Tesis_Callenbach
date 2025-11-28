import numpy as np

def calcular_static_dynamic(df, freq=14.45354719, window_seconds=10):
    window_size = int(freq * window_seconds)
    df['sax'] = df['ax'].rolling(window=window_size, center=True, min_periods=1).mean()
    df['say'] = df['ay'].rolling(window=window_size, center=True, min_periods=1).mean()
    df['saz'] = df['az'].rolling(window=window_size, center=True, min_periods=1).mean()
    df['sgx'] = df['gx'].rolling(window=window_size, center=True, min_periods=1).mean()
    df['sgy'] = df['gy'].rolling(window=window_size, center=True, min_periods=1).mean()
    df['sgz'] = df['gz'].rolling(window=window_size, center=True, min_periods=1).mean()
    df['ax'] = np.abs(df['ax'] - df['sax'])
    df['ay'] = np.abs(df['ay'] - df['say'])
    df['az'] = np.abs(df['az'] - df['saz'])
    df['gx'] = np.abs(df['gx'] - df['sgx'])
    df['gy'] = np.abs(df['gy'] - df['sgy'])
    df['gz'] = np.abs(df['gz'] - df['sgz'])
    return df

def encode_labels(labels, diccionario_codificacion):
    return np.array([diccionario_codificacion[str(label)] for label in labels])