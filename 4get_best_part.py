shots = [1, 9, 22, 34, 43, 51, 56, 74, 82]
hotest_danmu = [60, 70]


# 并集
def shot_or_damu(shots=[], danmu=[]):
    result = []
    for i in danmu:
        for s, e in zip(shots[:-1], shots[1:]):
            if s <= i <= e:
                result.append((s, e))
                break
    return [result[0][0], result[1][1]]


if __name__ == '__main__':
    print(shot_or_damu(shots, hotest_danmu))
