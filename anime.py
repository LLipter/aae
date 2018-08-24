import urllib.request
from bs4 import BeautifulSoup
import os
import time
import threading

# please delete the following 2 lines if you are not using MacOS
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

base_url = 'http://konachan.net/post?page='
dirname = 'anime'
headers = {'User-Agent': 'User-Agent:Mozilla/5.0'}

def getHtml(url):
    req = urllib.request.Request(url, headers=headers)
    html = urllib.request.urlopen(req).read()
    return html

def download_image(img_url,location):
    urllib.request.urlretrieve(img_url,location)

def download_image_in_page(page_no):
    try:
        print('start downloading page %d ...' % page_no)
        url = base_url + str(page_no)
        html = getHtml(url)
        soup = BeautifulSoup(html, 'html.parser')
        for img in soup.find_all('img', class_="preview"):
            # print(img['src'])
            filename = os.path.join(dirname, img['src'].split('/')[-1])
            download_image(img['src'], filename)
    except Exception as e:
        print('page %d download failed' % page_no)
        print(str(e))
        with open('failed_page_no.txt', 'a') as file:
            file.write('%d\n' % page_no)


def main():
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    start_time = time.time()
    i = 1
    page_num = 10000
    while i <= page_num:
        if(threading.activeCount() < 50):
            thd = threading.Thread(target=download_image_in_page,args=(i,))
            i += 1
            thd.start()
        time.sleep(1)
    for thd in threading.enumerate():
        if(thd.getName() != threading.currentThread().getName()):
            thd.join()
    stop_time = time.time()
    diff = stop_time - start_time
    print(diff,"seconds")
    print(diff/60,"minutes")
    print(diff/60/60,"hours")



if __name__ == '__main__':
    main()

