import numpy as np
x = np.array(
    [range(1,4),
     range(0,3)]
)

y = np.array(
    [[10, 11],
     [12, 13],
     [14, 15]]
).reshape(2,3)

print(np.concatenate([x, y]))
print(np.concatenate([x, y], 1))