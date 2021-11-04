def mat_mul(a, b):
    i, j = len(a), len(a[0])
    n, k = len(b), len(b[0])

    assert n == j
    temp = [[None for _ in range(k)] for _ in range(i)]
    for w in range(i):
        for h in range(k):
            temp[w][h] = 0
            for r in range(j):
                temp[w][h] += a[w][r] * b[r][h]
    return temp


def mat_add(a, b):
    i, j = len(a), len(a[0])
    n, k = len(b), len(b[0])

    assert i == n and j == k
    temp = [[None for _ in range(j)] for _ in range(i)]
    for w in range(i):
        for h in range(j):
            temp[w][h] = a[w][h] + b[w][h]
    return temp


def element_wise_mul(a, b):
    i, j = len(a), len(a[0])
    n, k = len(b), len(b[0])

    assert i == n and j == k
    temp = [[None for _ in range(j)] for _ in range(i)]
    for w in range(i):
        for h in range(j):
            temp[w][h] = a[w][h] * b[w][h]
    return temp


def transpose(a):
    i, j = len(a), len(a[0])
    temp = [[None for _ in range(i)] for _ in range(j)]
    for w in range(j):
        for h in range(i):
            temp[w][h] = a[h][w]
    return temp


def rescale(a, scale):
    i, j = len(a), len(a[0])
    temp = [[None for _ in range(j)] for _ in range(i)]
    for w in range(i):
        for h in range(j):
            temp[w][h] = a[w][h] * scale
    return temp


if __name__ == "__main__":
    a = [[1, 2], [3, 4]]
    b = [[5, 6], [7, 8]]
    print(mat_mul(a, b))
    print(mat_add(a, b))
    print(element_wise_mul(a, b))
    print(transpose(a))
    print(rescale(a, 1/5))
