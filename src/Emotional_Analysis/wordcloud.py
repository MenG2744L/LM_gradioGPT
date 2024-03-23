import jieba
import re
from collections import Counter

from pyecharts.charts import WordCloud


def part(filename: str):
    cut_words = ""
    for l in open(filename, encoding='utf-8'):
        l.strip('\n')
        l = re.sub("[A-Z a-z 0-9 \：\·\—\，\。\“ \”\！\【\】\…]", "", l)
        seg_list = jieba.cut(l, cut_all=False)
        cut_words += (" ".join(seg_list))
    all_words = cut_words.split()
    c = Counter()
    for x in all_words:
        if len(x) > 1 and x != '\n':
            c[x] += 1
    return c.most_common(50)


def wordcloud(result, shape: str):
    match shape:
        case '𒊹':
            shape = "circle"
        case '❤':
            shape = "cardioid"
        case '♦':
            shape = "diamond"
        case '▶':
            shape = "triangle-forward"
        case '▲':
            shape = "pentagon"
        case '★':
            shape = "star"
    word_cloud = WordCloud()
    word_cloud.add(series_name="热点分析",
                   data_pair=result,
                   word_size_range=[12, 60],
                   shape=shape
                   )

    return word_cloud
