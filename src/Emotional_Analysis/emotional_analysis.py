from snownlp import SnowNLP
from snapshot_selenium import snapshot
from pyecharts.render import make_snapshot

from src.Emotional_Analysis import wordcloud


def emotional_analysis_csv(file_path, shape):
    t = 0    # 初始化变量t，用于累加情感分析得分
    # 打开指定的CSV文件，并逐行读取内容
    for l in open(file_path, encoding='utf-8'):
        l.strip('\n')    # 去除每一行末尾的换行符
        t += SnowNLP(l).sentiments    # 使用SnowNLP库对处理后的文本行进行情感分析，并将得分累加到t中
    average = (t / len(open(file_path, encoding='utf-8').readlines()))    # 再次打开文件并计算行数，以便计算情感分析的平均分
    # 根据平均分判断情感倾向并给出结果描述
    if average >= 0.9:
        result = "人们一致好评！😘"
    elif 0.9 > average >= 0.75:
        result = "人们的评价不错！👍"
    elif 0.75 > average >= 0.6:
        result = "人们的评价比较一般！👌"
    else:
        result = "人们的评价很差！🆘"
    results = wordcloud.part(file_path)    # 调用wordcloud库的part函数处理文件，生成词云相关数据
    img_path = "E:\python-prj\gradioGPT-main\\assets\词云图.png"    # 指定词云图的保存路径
    # 使用snapshot-selenium渲染成图片，图案是用户个人选择的shape
    make_snapshot(snapshot, wordcloud.wordcloud(results, shape).render(), output_name=img_path)  # 保存为图片
    # 返回情感分析的平均分数（百分制）和对应的描述，以及词云图的保存路径
    return f"人们的评价的情感分析平均数值为：{average * 100:.2f}（百分制）,因此认为：" + result, img_path


# 定义一个函数 emotional_analysis_text，接收一个参数 text，用于进行文本情感分析
def emotional_analysis_text(text):
    # SnowNLP模型分析文本情感
    point = SnowNLP(text).sentiments
    # 判断情感倾向的分数
    # 如果情感倾向分数大于等于0.9
    if point >= 0.9:
        result = "这句话非常正面，充满了积极的能量！😘"
    # 如果情感倾向分数小于0.9大于等于0.75
    elif 0.9 > point >= 0.75:
        result = "这句话中蕴含着一些积极的情感，让人感到愉快！👍"
    # 如果情感倾向分数小于0.75大于等于0.6之间
    elif 0.75 > point >= 0.6:
        result = "这句话的情感倾向是中性的，没有明显的积极或消极情感！👌"
    # 如果情感倾向分数小于0.6
    else:
        result = "这句话传达了一种不太愉快或沮丧的心情！🆘"
    # 返回情感分析的结果，包括情感倾向的分数（百分制）和对应的情感描述
    return f"这句话情感分析数值为：{point * 100:.2f}（百分制）,因此认为" + result
