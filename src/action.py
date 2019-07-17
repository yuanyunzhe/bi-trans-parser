import numpy as np
import pickle
import matplotlib.pyplot as plt



with open("action.save", "rb") as f:
    a, b, c = pickle.load(f)

# lab = list(range(1, len(a) + 1))
#
# x = list(range(len(a)))
# total_width, n = 0.8, 2
# width = total_width / n
#
# plt.bar(x, a, width=width, fc="y")
# for i in range(len(x)):
#         x[i] += width
# plt.bar(x, b, width=width, tick_label=lab, fc="r")
# for i in range(len(x)):
#         x[i] += width
# plt.bar(x, c, width=width)

# aa = plt.hist(a, bins=30, range=(0.5, 30.5), color="r", alpha=0.5)
# bb = plt.hist(b, bins=30, range=(0.5, 30.5), color="b", alpha=0.5)

bins = np.linspace(-0.025, 1.025, 21)
print(len(a), len(b))
# aa = plt.hist(a, bins, color="r", edgecolor="black", alpha=0.5, label="l2r")
bb = plt.hist(b, bins, color="b", edgecolor="black", alpha=0.5, label="r2l")
cc = plt.hist(c, bins, color="g", edgecolor="black", alpha=0.5, label="do")

# print(a, np.mean(a), np.std(a))
# print(b, np.mean(b), np.std(b))
# print(len(a))

plt.xlabel("Position (Normalized)")
plt.ylabel("#Error")
plt.legend()
# plt.show()
plt.savefig("action_r2l.png")