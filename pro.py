import numpy as np

# Sample 1D array
def create_array():
    return np.array([3, 7, 2, 9, 4, 5, 1, 8, 6])

# 1. Basic Statistics
arr = create_array()
print("Array:", arr)
print("Sum:", arr.sum())
print("Mean:", arr.mean())
print("Min:", arr.min())
print("Max:", arr.max())
print("Std Dev:", arr.std())

# 2. Element-wise Arithmetic
print("\nElement-wise operations:")
print("+10:", arr + 10)
print("-2:", arr - 2)
print("*3:", arr * 3)
print("/2:", arr / 2)

other = np.arange(1, len(arr) + 1)
print("+ other:", arr + other)
print("* other:", arr * other)

# 3. Indexing & Slicing
print("\nIndexing & Slicing:")
print("First element:", arr[0])
print("Last two:", arr[-2:])
print("Middle slice [2:5]:", arr[2:5])
print("Every second element:", arr[::2])

# 4. Boolean Masking
mask = arr > 5
print("\nMask >5:", mask)
print("Filtered >5:", arr[mask])

# 5. Reshaping & Transpose
mat = arr.reshape(3, 3)
print("\nReshaped to 3x3:\n", mat)
print("Transposed:\n", mat.T)

# 6. Aggregations Along Axes
M = np.array([[1,2,3],[4,5,6],[7,8,9]])
print("\nMatrix M:\n", M)
print("Column sums:", M.sum(axis=0))
print("Row means:", M.mean(axis=1))

# 7. Stacking & Splitting
print("\nStacking & Splitting:")
a = np.array([1,2,3])
b = np.array([4,5,6])
print("hstack:", np.hstack((a,b)))
print("vstack:\n", np.vstack((a,b)))
print("Split into 3 parts:", np.array_split(arr, 3))

# 8. Linear Algebra
A = np.array([[1,2],[3,4]])
B = np.array([[5,6],[7,8]])
print("\nDot product A·B:\n", A.dot(B))
print("Inverse of A:\n", np.linalg.inv(A))
print("Determinant of A:", np.linalg.det(A))

# 9. Random Arrays
rng = np.random.default_rng(42)
print("\nRandom integers:", rng.integers(0, 10, 5))
print("Random floats:", rng.random(5))

# 10. Broadcasting
row = np.array([10,20,30])
print("\nBroadcasting M + row:\n", M + row)

# Exercise examples
# Z-score
z_scores = (arr - arr.mean()) / arr.std()
print("\nZ-scores:", np.round(z_scores, 2))

# Identity with replaced diagonal
I = np.eye(4)
np.fill_diagonal(I, np.arange(1,5))
print("\nCustom identity:\n", I)

# Zero out below diagonal
rand5 = rng.random((5,5))
mask_lower = np.tril_indices(5, k=-1)
rand5[mask_lower] = 0
print("\nZeroed lower triangle:\n", rand5)

# Replace odd with -1
arr2 = arr.copy()
arr2[arr2 % 2 == 1] = -1
print("\nOdds replaced:", arr2)

# Verify AB != BA
AB = A.dot(B)
BA = B.dot(A)
print("\nAB equals BA?", np.allclose(AB, BA))
