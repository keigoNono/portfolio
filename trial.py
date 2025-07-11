import matplotlib.pyplot as plt
label = ["A", "B", "C", "D"]
num = [20, 17, 25, 9]
plt.bar(label, num)
plt.savefig('./bar.png')