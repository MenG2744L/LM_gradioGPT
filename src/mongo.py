import pymongo

client = pymongo.MongoClient('localhost', 27017)
db = client["historyDB"]
history_db = db["history"]
roles=["基础GPT", "情感机器人", "心理医师", "励志教练", "音乐推荐"]

def insert_mongo(role: str, user_input: str, chatbot_output:str):
    history_db.insert_one({'role': role,
                           'user_input': user_input,
                           'chatbot_output': chatbot_output})

def default_role():
    role_counts = {role: 0 for role in roles}
    # 遍历每个角色，并计算其在数据库中的出现次数
    for role in roles:
        role_counts[role] = history_db.count_documents({"role": role})
    # 找出数据条数最多的角色
    max_role = max(role_counts, key=role_counts.get)

    return max_role

