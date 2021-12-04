from nltk import data as nltk_data
import numpy as np
import pandas as pd
from paths import *
from src.common import *


if __name__ == '__main__':
    print('Processing...\n')
    df = pd.read_csv(DB_DIR / RAW_DIR_NAME / DATA_FILENAME, encoding='utf-8')

    print('-- POSTS --')
    print('Unique authors:')
    print(df[AUTHOR].nunique())
    print('')

    print('Gender (unique authors):')
    print(df.groupby(GENDER).agg({AUTHOR: 'nunique'}).reset_index())
    print('')

    print('Gender (total, not unique):')
    print(df[GENDER].value_counts())
    print('')

    print('-- SENTENCES --')
    nltk_tokenize = nltk_data.load('tokenizers/punkt/english.pickle')
    male = df.loc[df['gender'] == 'male', ['txt']]
    female = df.loc[df['gender'] == 'female', ['txt']]
    # male = male.apply(lambda row: row.str.split('\r\n'), axis=1).explode('txt').reset_index(drop=True).apply(lambda row: row.str.strip()).replace('', np.nan).dropna()
    # female = female.apply(lambda row: row.str.split('\r\n'), axis=1).explode('txt').reset_index(drop=True).apply(lambda row: row.str.strip()).replace('', np.nan).dropna()
    # male = male.apply(lambda row: nltk_tokenize.tokenize(str(row)), axis=1).explode('txt').reset_index(drop=True).apply(lambda row: row.strip()).replace('', np.nan).dropna()
    male = male.apply(lambda row: nltk_tokenize.tokenize(row['txt']), axis=1).explode('txt').apply(lambda row: row.strip()).replace('', np.nan).dropna().to_frame('txt').reset_index(drop=True)
    female = female.apply(lambda row: nltk_tokenize.tokenize(row['txt']), axis=1).explode('txt').apply(lambda row: row.strip()).replace('', np.nan).dropna().to_frame('txt').reset_index(drop=True)
    print(f'Male: {len(male)}')
    print(f'Female: {len(female)}')
    print('')

    print('Average sentence length (in words)')
    print(f'Male: {np.mean(male.txt.str.split().apply(len))}')
    print(f'Female: {np.mean(female.txt.str.split().apply(len))}')
    print('')

    print('Median sentence length (in words)')
    print(f'Male: {np.median(male.txt.str.split().apply(len))}')
    print(f'Female: {np.median(female.txt.str.split().apply(len))}')
    print('')

    print('85% quantile sentence length (in words)')
    print(f'Male: {np.quantile(male.txt.str.split().apply(len), q=0.85)}')
    print(f'Female: {np.quantile(female.txt.str.split().apply(len), q=0.85)}')
    print('')

    print('Sentences >= 10 tokens')
    print(f'Male: {len(male[male.txt.str.split().apply(len) >= 10])}')
    print(f'Female: {len(female[female.txt.str.split().apply(len) >= 10])}')
