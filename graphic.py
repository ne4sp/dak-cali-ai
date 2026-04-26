from DatasetGenerator import Train, Line
import matplotlib.pyplot as plt

log1 = Train("logs/0723.log")
for line in log1.lines:
    plt.plot(line.lv)
plt.show()