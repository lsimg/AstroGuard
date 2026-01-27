import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

def load_real_nasa_data(data_dir):
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Папка не найдена: {data_dir}")

    files = sorted([f for f in os.listdir(data_dir) if not f.startswith('.')])
    data_list = []

    for filename in tqdm(files):
        filepath = os.path.join(data_dir, filename)
        try:
            df_temp = pd.read_csv(filepath, sep='\t', header=None)
            rms = np.sqrt((df_temp ** 2).mean())
            data_list.append(rms.values)
        except Exception as e:
            print(f"Пропуск файла {filename}: {e}")

    df_final = pd.DataFrame(data_list, columns=['Bearing1', 'Bearing2', 'Bearing3', 'Bearing4'])
    df_final.reset_index(inplace=True)
    df_final.rename(columns={'index': 'Time_Step'}, inplace=True)
    return df_final

my_folder_path = r'D:\Spacecraft Fault Prediction\IMS\IMS\2nd_test\2nd_test'

df = load_real_nasa_data(my_folder_path)

train_cols = ['Bearing1', 'Bearing2', 'Bearing3', 'Bearing4']
data_values = df[train_cols].values

scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(data_values)

train_data = x_scaled[:500]
test_data = x_scaled

input_dim = train_data.shape[1]

input_layer = Input(shape=(input_dim,))
encoder = Dense(2, activation="relu")(input_layer)
decoder = Dense(input_dim, activation="sigmoid")(encoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer='adam', loss='mse')

history = autoencoder.fit(
    train_data, train_data,
    epochs=100,
    batch_size=10,
    shuffle=True,
    verbose=0
)

reconstructions = autoencoder.predict(test_data)
mse = np.mean(np.power(test_data - reconstructions, 2), axis=1)

plt.figure(figsize=(12, 6))
plt.plot(mse, label='Индекс поломки (MSE)', color='blue')
threshold = np.max(mse[:500]) * 1.5
plt.axhline(y=threshold, color='r', linestyle='--', label='Порог тревоги')

plt.title('Предиктивная аналитика: Тест NASA (Bearing 1 Failure)')
plt.xlabel('Время (файлы/условные часы)')
plt.ylabel('Уровень аномалии')
plt.legend()
plt.grid(True)
plt.show()

print(f"Порог тревоги установлен на {threshold:.4f}")