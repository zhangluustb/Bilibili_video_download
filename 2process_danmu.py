import sys
import seaborn as sns
import matplotlib.pyplot as plt
import re


def text2list(text):
    l = re.findall('<d p=\".*?,', text)
    # print(l)
    l = [float(ll.split("\"")[-1][:-1]) for ll in l]
    return l


def get_hotest_part(danmu_path='bilibili_video/【朱一旦】55 来到了西湖，不禁诗性大发，写了一首经典.txt'):
    with open(danmu_path, 'r') as f:
        text = f.readlines()
    # print(text)
    assert len(text) == 1
    l = text2list(text[0])
    # split 10s 的片段
    # sns.distplot(l, bins=int(max(l) / 10))
    n, bins, patches = plt.hist(l, int(max(l) / 10), normed=1, facecolor='blue', alpha=0.5)
    index = n[1:-1].argmax() + 1
    print(bins[index], '~', bins[index + 1])
    plt.show()
    return [bins[index] // 10 * 10, bins[index + 1] // 10 * 10]


if __name__ == '__main__':
    # path = sys.argv[1]
    path = 'bilibili_video/【朱一旦】55 来到了西湖，不禁诗性大发，写了一首经典.txt'
    print(get_hotest_part(path))
