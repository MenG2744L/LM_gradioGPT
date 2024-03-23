from selenium import webdriver
import time
import pandas as pd
from selenium.webdriver.common.by import By


driver = webdriver.Chrome(executable_path="E:\python-prj\gradioGPT-main\chromedriver.exe")

driver.get('https://you.ctrip.com/sight/hangzhou14/49894.html')

j = 1
comment_list = []
for i in range(0, 300):  # 爬取300页的评论。
    driver.execute_script("window.scrollTo(0,document.body.scrollHeight);")  # 下滑到页面底端
    comments = driver.find_elements(
        By.CSS_SELECTOR,
        "div.commentDetail"
    )

    # comments = driver.find_elements_by_css_selector('div.commentDetail')  # 定位到commentDetail节点，爬取当前页的全部评论。
    for comment in comments:
        comment_list.append(comment.text)

    driver.execute_script("arguments[0].click();",
                          driver.find_element(
                              By.CLASS_NAME,
                              "ant-pagination-next"))  # 实现翻页功能，定位到ant-pagination-next节点，单击实现翻页。
    print('正在爬取第', i + 1, '页')
    # 每爬取十页保存一次
    if j % 10 == 0:
        comment_dataframe = pd.DataFrame(comment_list)   # 利用pandas将列表转换成dataframe类型
        # 保存爬取的评论为csv格式
        comment_dataframe.to_csv('西湖.csv', header=False, encoding='utf-8', index=False, mode='a')
        print("10页的评论已爬取成功！")
        comment_list = []
    j += 1
    time.sleep(2)  # 休眠2秒.

print("爬取完成！")
driver.quit()


