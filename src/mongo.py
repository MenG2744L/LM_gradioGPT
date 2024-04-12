import pymongo

client = pymongo.MongoClient('localhost', 27017)
db = client["historyDB"]
history_db = db["history"]

def insert_mongo(role: str, user_input: str):
    history_db.insert_one({'role': role,
                           'user_input': user_input})

