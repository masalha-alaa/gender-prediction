"""
Fetch data from reddit.com
"""
import re
from facebook_scraper import get_posts,enable_logging, get_profile
from requests import exceptions
from datetime import datetime
import time
import argparse
from enum import Enum
from paths import *
from src.common import *


DEBUG = False


def request_url_callback(url):
    global log_file
    print_log(f'Requesting page from:\n{url}', log_file)


def write_data(txt, url, author, gender, path):
    with open(path, 'a', encoding='utf-8') as f:
        if gender.lower() in ['male', 'female']:
            print(f'--{gender}--')
            if not DEBUG:
                txt = txt.replace('"', '""')
                print(f'"{txt}","{url}","{author}","{gender.lower()}"', file=f)


if __name__ == '__main__':
    ts_begin = datetime.now()
    ts_checkpoint = ts_begin
    print_log(f'Program started\n{ts_begin}')

    # output path
    main_output = ROOT_DIR / f'dataset {ts_begin.strftime(DATE_STR_SHORT)}'
    raw_output = main_output / RAW_DIR_NAME
    main_output.mkdir(exist_ok=True)
    raw_output.mkdir(exist_ok=True)

    data_path = raw_output / DATA_FILENAME
    log_file = main_output / f'log {ts_begin.strftime(DATE_STR_SHORT)}{LOG_EXT}'

    print_log(f'Program started\n{ts_begin}', log_file, to_console=False)

    # groups_d = {'Running the World': '360890597316575'}
    # groups_d = {'South America Backpacking / Traveling': '764281100331807'}
    groups_d = {'BBC News': 'bbcnews'}
    groups = list(groups_d.values())

    # collect data from facebook groups
    count = 0
    for i, group in enumerate(groups):
        print_log(f"Parsing FB group '{group}'...", log_file)
        for post in get_posts(group, pages=None, extra_info=True, timeout=60,
                              request_url_callback=request_url_callback,
                              # start_url=start_url,
                              cookies=str(CREDENTIALS_DIR / "cookies.json"),
                              options={"comments": True}):
            try:
                if post.get('user_id'):
                    gender = get_profile(post['user_id'], cookies=str(CREDENTIALS_DIR / 'cookies.json'))['Basic Info'].split()[0]
                    if gender:
                        txt = post.get('post_text') or post.get('text')
                        if txt:
                            try:
                                user_url = post.get('user_url').split('?')[0]
                            except:
                                user_url = post['user_id']
                            print('Post: ', end=' ')
                            write_data(txt, post.get('post_url'), user_url, gender, data_path)
                            count += 1
            except (KeyError, exceptions.ReadTimeout) as e:
                print('POST ERROR: ' + str(e))

            try:
                if post.get('comments_full'):
                    for comment in post['comments_full']:
                        if comment.get('commenter_url'):
                            user_id_reg_match = re.search(r"(?<=id=)\d+(?=&)", comment['commenter_url'])
                            if not user_id_reg_match:
                                user_id_reg_match = re.search(r"(?<=\.com/).+(?=\?)", comment['commenter_url'])
                            gender = get_profile(user_id_reg_match[0], cookies=str(CREDENTIALS_DIR / 'cookies.json'))['Basic Info'].split()[0]
                            if gender:
                                txt = comment.get('comment_text')
                                if txt:
                                    user_url = comment.get('commenter_url').split('?')[0]
                                    print('Comment: ', end=' ')
                                    write_data(txt, post.get('comment_url'), user_url, gender, data_path)
                                    count += 1
            except (KeyError, exceptions.ReadTimeout) as e:
                print('COMMENT ERROR: ' + str(e))

    print_log(f'\nCollected {count} posts\n'
              f'{datetime.now()}\n'
              f'TOTAL TIME: {datetime.now() - ts_begin}', log_file)
