import pymysql


def open_login_window(input_name: str, input_password: str) -> bool:
    # 创建游标对象
    conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='123456', db='mysql')
    cursor = conn.cursor()
    sql = """SELECT NAME, PASSWORD FROM login"""  # sql查询语句，查找login表中name和password两个值
    cursor.execute(sql)  # 执行查询语句
    results = cursor.fetchall()
    login_success = False
    for temp in results:
        if input_name == temp[0] and input_password == temp[1]:
            login_success = True
            break
    cursor.close()  # 关闭游标
    conn.close()
    return login_success
