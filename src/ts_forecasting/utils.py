import pandas as pd
from copy import deepcopy as dc
from sklearn.preprocessing import MinMaxScaler


def prepare_dataframe_for_lstm(df, n_steps):
    df = dc(df)

    df.set_index('Date', inplace=True)

    for i in range(1, n_steps+1):
        df[f'Close(t-{i})'] = df['Close'].shift(i)

    df.dropna(inplace=True)

    return df


def df_preprocessing():
    # Hyperparams
    lookback = 7

    # Read the df
    data = pd.read_csv('data/ts-frc/data.csv')
    print(data.head())

    # Preprocess the df
    data = data[['Date', 'Close']]
    data['Date'] = pd.to_datetime(data['Date'])
    shifted_df = prepare_dataframe_for_lstm(data, lookback)
    shifted_df_as_np = shifted_df.to_numpy()

    # Preprocessing
    scaler = MinMaxScaler(feature_range=(-1, 1))
    shifted_df_as_np = scaler.fit_transform(shifted_df_as_np)

    X = shifted_df_as_np[:, 1:]
    y = shifted_df_as_np[:, 0]

    split_index = int(len(X) * 0.95)

    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    X_train = X_train.reshape((-1, lookback, 1))
    X_test = X_test.reshape((-1, lookback, 1))
    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))

    return X_train, X_test, y_train, y_test
