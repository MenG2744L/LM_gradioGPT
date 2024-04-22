from transformers import pipeline


# 定义音频情绪分析函数，接收音频文件路径作为参数，返回情绪分析结果的字典
def audio_sentiment(path) -> dict:
    # 使用Hugging Face的pipeline函数创建一个音频分类的pipeline，并指定模型为wav2vec2-lg-xlsr-en-speech-emotion-recognition
    audio_to_list = pipeline('audio-classification', model='ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition')
    result = audio_to_list(path)  # 使用创建的pipeline对音频文件进行处理，返回情绪分类的结果
    output_dict = {}  # 初始化一个空字典，用于存储情绪分析的结果
    # 遍历情绪分类的结果
    for item in result:
        label = item['label']  # 提取情绪标签
        score = item['score']  # 提取对应情绪标签的分数
        output_dict[label] = score  # 将情绪标签和分数作为键值对存储到输出字典中
    return output_dict


# 定义图片人物情绪分析函数，接收图片文件路径作为参数，返回情绪分析结果的字典
def img_sentiment(path) -> dict:
    # 使用Hugging Face的pipeline函数创建一个图像分类的pipeline，并指定模型为vit-base-patch16-224-finetuned-Visual-Emotional
    img_to_list = pipeline('image-classification', model='ntnxx2/vit-base-patch16-224-finetuned-Visual-Emotional')
    result = img_to_list(path)  # 使用创建的pipeline对图片文件进行处理，返回情绪分类的结果
    output_dict = {}  # 初始化一个空字典，用于存储情绪分析的结果
    # 遍历情绪分类的结果
    for item in result:
        label = item['label']  # 提取情绪标签
        score = item['score']  # 提取对应情绪标签的分数
        output_dict[label] = score  # 将情绪标签和分数作为键值对存储到输出字典中
    return output_dict
