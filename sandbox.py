import numpy as np
import matplotlib.pyplot as plt

# Параметры кривой (например, циклоида)
t = np.linspace(0, 2 * np.pi, 1000)
x = 2 * (t - np.sin(t))
y = 2 * (1 - np.cos(t))

# Вычисление центра масс (для однородной арки циклоиды)
x_c = np.pi * 2
y_c = (4 * 2) / 3

# Визуализация
plt.figure(figsize=(8, 5))
plt.plot(x, y, label='Кривая (арка циклоиды)', color='blue', linewidth=2.5)
plt.scatter(x_c, y_c, color='red', s=100, zorder=5, label=f'Центр масс ({x_c:.2f}, {y_c:.2f})')
plt.title('Визуализация центра масс кривой')
plt.xlabel('x')
plt.ylabel('y')
plt.axhline(0, color='black', linewidth=0.8)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()
