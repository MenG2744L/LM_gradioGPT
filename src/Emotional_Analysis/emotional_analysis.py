from snownlp import SnowNLP
from snapshot_selenium import snapshot
from pyecharts.render import make_snapshot

from src.Emotional_Analysis import wordcloud


def emotional_analysis_csv(file_path, shape):
    t = 0
    for l in open(file_path, encoding='utf-8'):
        l.strip('\n')
        t += SnowNLP(l).sentiments
    average = (t / len(open(file_path, encoding='utf-8').readlines()))
    if average >= 0.9:
        result = "人们一致好评！😘"
    elif 0.9 > average >= 0.75:
        result = "人们的评价不错！👍"
    elif 0.75 > average > 0.5:
        result = "人们的评价比较一般！👌"
    else:
        result = "人们的评价很差！🆘"
    results = wordcloud.part(file_path)
    img_path = "E:\python-prj\gradioGPT-main\\assets\词云图.png"
    # 使用snapshot-selenium渲染成图片
    make_snapshot(snapshot, wordcloud.wordcloud(results, shape).render(), output_name=img_path)  # 保存为图片
    return f"人们的评价的情感分析平均数值为：{average * 100:.2f}（百分制）,因此认为：" + result, img_path


def emotional_analysis_text(text):
    # SnowNLP模型分析文本情感
    point = SnowNLP(text).sentiments
    if point >= 0.9:
        result = "这句话非常正面，充满了积极的能量！😘"
    elif 0.9 > point >= 0.75:
        result = "这句话中蕴含着一些积极的情感，让人感到愉快！👍"
    elif 0.75 > point > 0.4:
        result = "这句话的情感倾向是中性的，没有明显的积极或消极情感！👌"
    else:
        result = "这句话传达了一种不太愉快或沮丧的情！🆘"
    return f"人们的评价的情感分析平均数值为：{point:.2f}（百分制）,因此认为" + result