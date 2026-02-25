"""01 — Arrays: the building block of every neural network."""

import mlx.core as mx

# A single number
a = mx.array(5.0)
print("a =", a, " shape:", a.shape)

# A row of numbers (a "vector")
b = mx.array([1.0, 2.0, 3.0])
print("b =", b, " shape:", b.shape)

# A grid of numbers (a "matrix")
c = mx.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0]])
print("c =")
print(c)
print("shape:", c.shape)

# dtype = how precise each number is (and how much memory it eats)
f32 = mx.array([1.0, 2.0], dtype=mx.float32)  # 4 bytes per number
f16 = mx.array([1.0, 2.0], dtype=mx.float16)  # 2 bytes per number

print("\nfloat32:", f32, " — full precision, 4 bytes each")
print("float16:", f16, " — half precision, 2 bytes each")
print("(A 7B model in float16 = 7 billion * 2 bytes = ~14 GB of memory)")
