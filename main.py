import requests
from openpyxl import workbook
from bs4 import BeautifulSoup as bSoup
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')
keyword = 'AMD'


def get_data(_src):
    html = requests.get(_src).content
    soup = bSoup(html, 'lxml')
    global ws
    g_name = []
    g_content = []
    content_list = soup.find_all("div", class_="snippet")
    name_list = soup.find_all("a", class_="title")
    for name in name_list:
        print('Find: '+name.text)
        g_name.append(name.text)
    for content in content_list:
        g_content.append(content.text)
    for i in range(len(g_name)):
        ws.append([g_name[i], g_content[i]])


if __name__ == '__main__':
    wb = workbook.Workbook()
    ws = wb.active
    ws.append(['name', 'info'])
    _src = 'https://www.bing.com/news/search?q='+keyword+'&setmkt=en-us&setlang=en-us'
    get_data(_src)
    wb.save('result.xlsx')
