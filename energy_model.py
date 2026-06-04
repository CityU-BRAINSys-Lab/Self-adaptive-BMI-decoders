import numpy as np
import math

import numpy as np
import matplotlib.pyplot as plt

# ===== Point =====
x = np.array([8, 32, 1024])
y = np.array([10/2, 20/2, 100/2])

# ===== Fit =====
a, b = np.polyfit(x, y, 1)

# ===== Generate fit line =====
x_line = np.linspace(min(x), max(x), 100)
y_line = a * x_line + b


# # ===== Figure =====
# plt.figure()
# plt.scatter(x, y, color='#065084')
# plt.plot(x_line, y_line, color="#255B82", label="Fitted energy line")

# for i in range(len(x)):
#     plt.text(x[i], y[i]+2, f"{x[i]}KB", fontsize=9, ha='center', va='top')
#     plt.text(x[i], y[i], f"{y[i]}pJ", fontsize=9, ha='center', va='top')
    

# plt.xlabel("Cache size (KB)")
# plt.ylabel("Energy (pJ)")
# # plt.title(f"y = {a:.4f}x + {b:.4f}")
# plt.legend()
# plt.savefig("./energy_model.jpg", dpi=300, bbox_inches='tight')
# plt.savefig("./energy_model.pdf", bbox_inches='tight')
# plt.show()

