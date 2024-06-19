import numpy as np

def df_to_X_y3(df, window_size,sub1,sub2):
  df_as_np = df.to_numpy()
  X = []
  y = []
  for i in range(len(df_as_np)-window_size):
    row = [r for r in df_as_np[i:i+window_size]]
    X.append(row)
    label = [df_as_np[i+window_size][sub1], df_as_np[i+window_size][sub2]]
    y.append(label)
  return np.array(X), np.array(y)

