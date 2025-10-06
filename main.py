import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from DatasetGenerator import generate, Train

fulldata = generate()

X = [data[0] for data in fulldata]
Y = [data[1] for data in fulldata]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.1, random_state=42
)

model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.1,
    loss_function='RMSE',
    random_seed=42,
    verbose=0
)

print("Начало обучения CatBoost...")
model.fit(X_train, Y_train)
print("Обучение завершено.")

Y_pred = model.predict(X_test)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)

print(f"\nСреднеквадратичная ошибка (RMSE) на тестовой выборке: {rmse:.4f}")

train = Train('trains/train_14.log')
for line in train.lines:
    line.to_compare = train.lines[0]
    line.create_o()
for line in train.lines:
    print('REAL', line.conc)
    print('AI', round(model.predict(line.execute_data()[0]), 1))