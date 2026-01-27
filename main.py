import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense


# --- 1. ФУНКЦИЯ ЗАГРУЗКИ ДАННЫХ NASA ---
def load_real_nasa_data(data_dir):
    # Проверяем, существует ли папка
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Папка не найдена: {data_dir}")

    files = sorted([f for f in os.listdir(data_dir) if not f.startswith('.')])
    data_list = []

    print(f"Чтение {len(files)} файлов из {data_dir}...")

    for filename in tqdm(files):
        filepath = os.path.join(data_dir, filename)
        try:
            # Читаем файл (4 колонки = 4 подшипника)
            df_temp = pd.read_csv(filepath, sep='\t', header=None)
            # Считаем RMS (среднеквадратичное значение)
            rms = np.sqrt((df_temp ** 2).mean())
            data_list.append(rms.values)
        except Exception as e:
            print(f"Пропуск файла {filename}: {e}")

    df_final = pd.DataFrame(data_list, columns=['Bearing1', 'Bearing2', 'Bearing3', 'Bearing4'])
    df_final.reset_index(inplace=True)
    df_final.rename(columns={'index': 'Time_Step'}, inplace=True)
    return df_final


# --- 2. НАСТРОЙКИ И ЗАГРУЗКА ---
# Ваш путь (исправленный)
# Добавили еще один \IMS в середину
my_folder_path = r'D:\Spacecraft Fault Prediction\IMS\IMS\2nd_test\2nd_test'

# Загружаем данные
print("Запуск загрузки...")
df = load_real_nasa_data(my_folder_path)
print("Размер данных:", df.shape)

# --- 3. ПОДГОТОВКА ДАННЫХ ДЛЯ ИИ ---
# Берем данные с 4 сенсоров (убираем колонку Time_Step)
train_cols = ['Bearing1', 'Bearing2', 'Bearing3', 'Bearing4']
data_values = df[train_cols].values

# Нормализация (приводим числа к диапазону 0-1)
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(data_values)

# РАЗДЕЛЕНИЕ:
# Первые 500 точек считаем "здоровым периодом" для обучения
train_data = x_scaled[:500]
test_data = x_scaled  # Тестируем на всем периоде

# --- 4. СОЗДАНИЕ И ОБУЧЕНИЕ МОДЕЛИ ---
input_dim = train_data.shape[1]  # 4 входа

input_layer = Input(shape=(input_dim,))
# Сжимаем информацию
encoder = Dense(2, activation="relu")(input_layer)
# Восстанавливаем
decoder = Dense(input_dim, activation="sigmoid")(encoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer='adam', loss='mse')

print("Обучение нейросети...")
history = autoencoder.fit(
    train_data, train_data,
    epochs=100,  # Чуть больше эпох для точности
    batch_size=10,
    shuffle=True,
    verbose=0
)

# --- 5. ПОИСК АНОМАЛИЙ И ГРАФИК ---
print("Анализ данных...")
reconstructions = autoencoder.predict(test_data)
# Ошибка = (Оригинал - Восстановленное)^2
mse = np.mean(np.power(test_data - reconstructions, 2), axis=1)

# Рисуем график
plt.figure(figsize=(12, 6))
plt.plot(mse, label='Индекс поломки (MSE)', color='blue')
# Рисуем порог тревоги (чуть выше среднего по тренировочным данным)
threshold = np.max(mse[:500]) * 1.5
plt.axhline(y=threshold, color='r', linestyle='--', label='Порог тревоги')

plt.title('Предиктивная аналитика: Тест NASA (Bearing 1 Failure)')
plt.xlabel('Время (файлы/условные часы)')
plt.ylabel('Уровень аномалии')
plt.legend()
plt.grid(True)
plt.show()

print(f"Готово! Порог тревоги установлен на {threshold:.4f}")