def mat_mul(a, b, i, j, n, k):
    assert n == j
    temp = [[None for _ in range(k)] for _ in range(i)]
    for w in range(i):
        for h in range(k):
            temp[w][h] = 0
            for r in range(j):
                temp[w][h] += a[w][r] * b[r][h]
    return temp


def mat_add(a, b, i, j, n, k):
    assert i == n and j == k
    temp = [[None for _ in range(j)] for _ in range(i)]
    for w in range(i):
        for h in range(j):
            temp[w][h] = a[w][h] + b[w][h]
    return temp


if __name__ == "__main__":
    a = [[1, 2], [3, 4]]
    b = [[5, 6], [7, 8]]
    print(mat_mul(a, b, 2, 2, 2, 2))
    print(mat_add(a, b, 2, 2, 2, 2))
