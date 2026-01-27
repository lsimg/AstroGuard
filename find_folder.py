import os

# Начинаем искать от той папки, где лежит скрипт
start_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Ищу папку '2nd_test' внутри: {start_dir}...")

found = False
# os.walk проходит по всем вложенным папкам
for root, dirs, files in os.walk(start_dir):
    if '2nd_test' in dirs:
        # Формируем полный путь
        full_path = os.path.join(root, '2nd_test')

        print("\n" + "=" * 40)
        print("✅ УРА! ПАПКА НАЙДЕНА!")
        print("Скопируйте строку ниже и вставьте её в main.py:")
        print("-" * 20)
        print(f"my_folder_path = r'{full_path}'")
        print("-" * 20)
        print("=" * 40 + "\n")
        found = True
        break

if not found:
    print("\n❌ Папка '2nd_test' не найдена.")
    print("Пожалуйста, убедитесь, что вы распаковали архив IMS.zip")
    print("Вы должны видеть файлы, если зайдете в папки через Проводник.")