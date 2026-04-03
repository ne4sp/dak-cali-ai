import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from DatasetGenerator import generate, Train
import matplotlib.pyplot as plt

fulldata = generate()

X = [data[0] for data in fulldata]
Y = [data[1] for data in fulldata]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.1, random_state=42
)


model = CatBoostRegressor(
    iterations=600,
    learning_rate=0.1,
    loss_function='RMSE',
    random_seed=42,
)


print("Начало обучения CatBoost...")
model.fit(X_train, Y_train)
print("Обучение завершено.")

Y_pred = model.predict(X_test)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)




train = Train('dak/train.log')
log = Train('dak/conc_history_2025-10-14.log')

COL_WIDTH = 10

# Вывод заголовков столбцов
print(f"{'REAL':^{COL_WIDTH}} | {'AI':^{COL_WIDTH}}")
print("-" * (2 * COL_WIDTH + 3))  # Разделительная линия

for line in log.lines:
    line.to_compare = train.lines[0]
    line.create_o()

real = 0
counter = 0
ai = 0

for line in log.lines:
    print(round(line.conc, 2), '-', abs(round(model.predict(line.execute_data()[0]), 2)))
    counter += 1
    real += round(line.conc, 1)
    ai += abs(round(model.predict(line.execute_data()[0]), 1))
print('avg ai = ', ai/counter)
print('avg real = ', real/counter)


