"""
Fetch data from reddit.com
"""

from psaw import PushshiftAPI
import praw
import psraw  # https://github.com/danthedaniel/psraw
from prawcore import exceptions
from datetime import datetime
import time
import argparse
from enum import Enum
from paths import *
from src.common import *


class FETCHING_LIB_ENUM(Enum):
    PRAW = 0
    PSRAW = 1
    PSAW = 2


DEBUG = False
LIBRARY = FETCHING_LIB_ENUM.PSAW


class PsawClass:
    api = None

    def __init__(self):
        self._initialized = False

    def initialize(self, r_instance):
        if not self._initialized:
            PsawClass.api = PushshiftAPI(r_instance)
            self._initialized = True

    @staticmethod
    def get_instance():
        return PsawClass.api


def submissions_gen(library, r_instance, subreddit_name):
    if library == FETCHING_LIB_ENUM.PRAW:
        return r_instance.subreddit(subreddit_name).hot(limit=None)
    elif library == FETCHING_LIB_ENUM.PSRAW:
        return psraw.submission_search(r_instance, q='', subreddit=subreddit_name, limit=10000)
    elif library == FETCHING_LIB_ENUM.PSAW:
        pInstance = PsawClass()
        pInstance.initialize(r_instance)
        start_epoch = int(datetime(2017, 1, 1).timestamp())
        api = pInstance.get_instance()
        return api.search_submissions(after=start_epoch, subreddit=subreddit_name)


def write_data(txt, submission, path, ignore_gender=False):
    def _char_to_gen(gender):
        male = ['male', '♂',
                'teens male', 'early 20s male',
                'late 20s male', 'early 30s male', 'late 30s male', '50s male',
                '60+ male']
        female = ['female', '♀',
                  'teens female', 'early 20s female',
                  'late 20s female', 'early 30s female', 'late 30s female', '50s female',
                  '60+ female',
                  'lesbian']
        gender = gender.lower() if gender else None
        if gender:
            return 'male' if gender in male else 'female' if gender in female else None
        if not ignore_gender:
            return None
    with open(path, 'a', encoding='utf-8') as f:
        url, author, gender = submission.permalink,\
                              submission.author.name if submission.author else '',\
                              submission.author_flair_text or (ignore_gender and '')
        converted_gender = _char_to_gen(gender)
        if converted_gender or ignore_gender:
            print(f'--{gender}--')
            if not DEBUG:
                txt = txt.replace('"', '""')
                print(f'"{txt}","{url}","{author}","{converted_gender}"', file=f)


if __name__ == '__main__':
    # -d mIXUAoEFXFNKzQ -s tlc3pqzTjj9VRd6UjC1tMKf2JBPn1w -a "Windows10 (alaa_137)"
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--id", required=True, help="Reddit client ID")
    ap.add_argument("-s", "--secret", required=True, help="Reddit client secret")
    ap.add_argument("-a", "--agent", required=True, help="User agent")
    args = vars(ap.parse_args())
    client_id = args['id']
    client_secret = args['secret']
    user_agent = args['agent']

    ts_begin = datetime.now()
    ts_checkpoint = ts_begin
    print_log(f'Program started\n{ts_begin}')
    print_log(f'Library:{LIBRARY}')

    # output path
    main_output = ROOT_DIR / f'dataset {ts_begin.strftime(DATE_STR_SHORT)}'
    raw_output = main_output / RAW_DIR_NAME
    main_output.mkdir(exist_ok=True)
    raw_output.mkdir(exist_ok=True)

    data_path = raw_output / DATA_FILENAME
    log_file = main_output / f'log {ts_begin.strftime(DATE_STR_SHORT)}{LOG_EXT}'

    print_log(f'Program started\n{ts_begin}', log_file, to_console=False)

    reddit = praw.Reddit(client_id=client_id,
                         client_secret=client_secret,
                         user_agent=user_agent)
    # subreddits = ['AskMen', 'AskWomen']
    # subreddits = ['AskMen']
    # subreddits = ['AskWomen']
    # subreddits = ['askwomenadvice']
    # subreddits = ['relationship_advice']
    subreddits = ['AskMenAdvice']
    # subreddits = ['relationships']

    untagged_subs = ['relationships']

    # collect data from subreddits
    count = 0
    for i, subreddit in enumerate(subreddits):
        print_log(f"Parsing subreddit '{subreddit}'...", log_file)
        for submission in submissions_gen(LIBRARY, reddit, subreddit):
            time.sleep(1)
            flair = submission.author_flair_text
            if flair or subreddit in untagged_subs:
                txt = ''
                if submission.title not in ['[removed]', '[deleted]']:
                    txt += submission.title
                if submission.selftext and submission.selftext not in ['[removed]', '[deleted]']:
                    if txt:
                        txt += '\n'
                    txt += submission.selftext
                if txt:
                    write_data(txt, submission, data_path, subreddit in untagged_subs)
                    count += 1

            # get ALL comments in ALL levels
            # if subreddit not in untagged_subs:
            #     try:
            #         for comment in submission.comments.list():
            #             flair = comment.author_flair_text if hasattr(comment, 'author_flair_text') else None
            #             if flair and comment.body not in ['[removed]', '[deleted]']:
            #                 write_data(comment.body, comment, data_path, subreddit in untagged_subs)
            #                 count += 1
            #     except exceptions.ServerError:
            #         print_log('* Server error', log_file)
            #         time.sleep(3)

            # get comments up to 2nd level
            if subreddit not in untagged_subs:
                try:
                    network_error = True
                    tries = 15
                    while network_error and tries:
                        try:
                            submission.comments.replace_more(limit=None)
                            network_error = False
                        except exceptions.RequestException as e:
                            print_log(f'{str(e)}.\nRetrying...', log_file)
                            tries -= 1
                            time.sleep(2)
                    for top_level_comment in submission.comments:
                        flair = top_level_comment.author_flair_text if hasattr(top_level_comment, 'author_flair_text') else None
                        if flair and top_level_comment.body not in ['[removed]', '[deleted]']:
                            write_data(top_level_comment.body, top_level_comment, data_path, subreddit in untagged_subs)
                            count += 1
                        for second_level_comment in top_level_comment.replies:
                            flair = second_level_comment.author_flair_text if hasattr(second_level_comment, 'author_flair_text') else None
                            if flair and second_level_comment.body not in ['[removed]', '[deleted]']:
                                write_data(second_level_comment.body, second_level_comment, data_path, subreddit in untagged_subs)
                                count += 1
                except exceptions.ServerError:
                    print_log('* Server error', log_file)
                    time.sleep(3)

    print_log(f'\nCollected {count} posts\n'
              f'{datetime.now()}\n'
              f'TOTAL TIME: {datetime.now() - ts_begin}', log_file)
