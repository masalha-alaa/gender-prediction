from collections import OrderedDict
from enum import Enum
import re
import emoji
from datetime import datetime
from src.my_random import SEED
from paths import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
import pandas as pd
from nltk import word_tokenize
from nltk import data as nltk_data, pos_tag
from nltk.sentiment import SentimentIntensityAnalyzer
from src.select_best import *
import scipy.sparse as sp
from multiprocessing import Pool, cpu_count


def clean_line_parallel(params):
    part, emoticons_ = params
    return part.apply(lambda row: clean_line(row, emoticons_))


def clean_line(line, emos=None):
    if not line:
        return ''
    EMOJI = r'emj'
    URL = r'url'
    remove = ['your comment has been removed', 'this comment or post has been removed', 'removed ', 'removed. ',
              'your submission has been remove', 'this post has been removed', 'your post has been removed']
    line = line.lower()
    if any(re.match(r, line) for r in remove):
        return ''
    line = emoji.get_emoji_regexp().sub(EMOJI, line)
    if emos is None:
        emoticons_ = emoticons
    else:
        emoticons_ = emos
    for emoticon in emoticons_:
        line = line.replace(emoticon, EMOJI)
    line = re.sub(r'\(?http\S+\)?', URL, line)
    return line


def posify(txt):
    return ' '.join([pair[1] for pair in pos_tag(txt.split())])


def tokenize(txt):
    # return ' '.join(word_tokenize(clean_line(txt)))
    # return clean_line(txt)
    return ' '.join(word_tokenize(txt))


def sentecize(txt):
    return tokenizer.tokenize(txt)


def aggregate(df, sentences_n=4):
    df_agg = df.groupby(df.index // sentences_n).agg({'txt': ' '.join, 'pos': ' '.join, 'gender': 'first'}).reset_index(drop=True).copy()
    return df_agg


def score(clf, train_vals, test_vals, train_lbls, test_lbls, ftr_lst, print_confidence_level=False):
    train_acc = clf.score(train_vals, train_lbls)
    test_acc = clf.score(test_vals, test_lbls)
    train_mse = mean_squared_error(train_lbls, clf.predict(train_vals))
    test_mse = mean_squared_error(test_lbls, clf.predict(test_vals))

    s_train = f'==> Train acc: {train_acc * 100:.2f}%'
    s_test = f'==> Test acc: {test_acc * 100:.2f}%'
    if print_confidence_level:
        confidence_interval_train = Z_CONFIDENCE.value * np.sqrt((train_acc * (1 - train_acc)) / len(train_lbls))
        confidence_interval_test = Z_CONFIDENCE.value * np.sqrt((test_acc * (1 - test_acc)) / len(test_lbls))
        s_train += f' (±{confidence_interval_train*100:.0f}%)'
        s_test += f' (±{confidence_interval_test*100:.0f}%)'

    s_train += f', MSE: {train_mse:.3f}'
    s_test += f', MSE: {test_mse:.3f}'

    if print_confidence_level:
        confidence_interval_train = Z_CONFIDENCE.value * np.sqrt((train_mse * (1 - train_mse)) / len(train_lbls))
        confidence_interval_test = Z_CONFIDENCE.value * np.sqrt((test_mse * (1 - test_mse)) / len(test_lbls))
        s_train += f' (±{confidence_interval_train:.3f}), {Z_CONFIDENCE.name}'
        s_test += f' (±{confidence_interval_test:.3f}), {Z_CONFIDENCE.name}'

    print(s_train)
    print(s_test)

    #  Features are most to least important
    if hasattr(clf, 'coef_'):
        print('Select K Best...')
        # best_features = select_k_best(clf.coef_, ftr_lst, SELECT_K_BEST)
        best_features = select_k_best(np.std(train_vals.values, 0) * clf.coef_, ftr_lst, SELECT_K_BEST)
        print(best_features)
    elif hasattr(clf, 'feature_importances_'):
        print('Select K Best...')
        feat_importances = pd.Series(clf.feature_importances_, index=ftr_lst)
        print(np.array(feat_importances.nlargest(SELECT_K_BEST).index.tolist()))

    print('')
    return train_acc, train_mse, test_acc, test_mse


class FeaturesEnum(Enum):
    TF_IDF_TXT = 1
    TF_IDF_POS = 2
    AVG_SENT_LEN = 3
    FW_WRDS = 4
    CUSTOM_WRDS = 5
    SENTIMENT_ANALYSIS = 6
    TF_IDF_TXT_CHAR = 7


class StandardDeviationZ(Enum):
    CONFIDENCE_LEVEL_90 = 1.64
    CONFIDENCE_LEVEL_95 = 1.96
    CONFIDENCE_LEVEL_98 = 2.33
    CONFIDENCE_LEVEL_99 = 2.58

    @property
    def name(self):
        return super().name.replace('_', ' ') + '%'


if __name__ == '__main__':
    ts = datetime.now()
    print('Program started')
    print(f'{ts}\n')
    print(f'Dataset: {DB_DIR}')

    EMOJI = r'emj'
    URL = r'url'
    MALE, FEMALE = 0, 1
    PRECUT = 0.3
    CLASS_SIZE = None  # None to take min max
    SELECT_K_BEST = 100
    # TEST_ON_ANOTHER_DS = EXPLICIT_DS("dataset3")  # None for testing on same DS
    # TEST_ON_ANOTHER_DS = EXPLICIT_DS("dataset1 and 2021-11-06 19-51-29")  # None for testing on same DS
    TEST_ON_ANOTHER_DS = None  # None for testing on same DS
    TRAIN_FRAC = 1 if TEST_ON_ANOTHER_DS else 0.70
    SECOND_DS_FRAC = 1.0  # applicable only if TEST_ON_ANOTHER_DS is not None
    MIN_SENT_LEN = 5  # in tokens  # TODO: Do some trials
    Z_CONFIDENCE = StandardDeviationZ.CONFIDENCE_LEVEL_95
    FEATURES = {FeaturesEnum.TF_IDF_TXT: True,
                FeaturesEnum.TF_IDF_TXT_CHAR: False,
                FeaturesEnum.TF_IDF_POS: True,
                FeaturesEnum.AVG_SENT_LEN: True,
                FeaturesEnum.FW_WRDS: False,
                FeaturesEnum.CUSTOM_WRDS: True,
                FeaturesEnum.SENTIMENT_ANALYSIS: True}

    # read dataset
    data = pd.read_csv(RAW_DATA_PATH)
    if EXPECTED_SIZE_DICT[DB_DIR] != data.shape[0]:
        print(f"* WARNING: Data size: {data.shape[0]:,}. Expected: {EXPECTED_SIZE_DICT[DB_DIR]:,}")
    if 0 < PRECUT < 1:
        data = data.sample(frac=PRECUT, random_state=SEED)

    data = data.drop(data[(data.gender != 'male') & (data.gender != 'female')].index).reset_index(drop=True)

    if TEST_ON_ANOTHER_DS:
        test_data = pd.read_csv(TEST_ON_ANOTHER_DS).sample(frac=SECOND_DS_FRAC, random_state=SEED).reset_index(drop=True)
        test_data = test_data[~test_data['url'].isin(data['url'])].reset_index(drop=True)  # Remove intersection between train and test datasets
        test_data['gender'].replace(['male', 'female'], [MALE, FEMALE], inplace=True)
        test_class_size = min(test_data[test_data['gender'] == MALE].shape[0], test_data[test_data['gender'] == FEMALE].shape[0])
        test_data = pd.concat([(test_data[test_data['gender'] == MALE]).sample(test_class_size, random_state=SEED),
                               (test_data[test_data['gender'] == FEMALE]).sample(test_class_size, random_state=SEED)], axis=0).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        data_size = len(data)
        data = pd.concat((data, test_data), axis=0).reset_index(drop=True)

    # Clean
    clean_ts = datetime.now()
    print('Cleaning and tokenzing...', end=' ')
    with open(EXTRA_DIR / EMOTICONS_LST_FILE) as f:
        emoticons = [emoticon.lower() for emoticon in list(set(f.read().split()))]
    if MIN_SENT_LEN > 1:
        # TODO: Ditch short before or after exploding?!
        data[data.txt.str.split().apply(len) < MIN_SENT_LEN] = ''
        # data = data[data.txt.str.split().apply(len) >= MIN_SENT_LEN].reset_index(drop=True)

    # PARALLEL BEGIN
    pools = cpu_count() // 2
    with Pool(pools) as pool:
        groups = [(part, emoticons) for part in np.array_split(data['txt'], pools)]
        data['txt'] = pd.concat(pool.map(clean_line_parallel, groups), axis=0)
        data['txt'] = pool.map(tokenize, data['txt'])
    # PARALLEL END
    # data['txt'] = data['txt'].apply(clean_line)  # Not parallel

    if TEST_ON_ANOTHER_DS:
        data, test_data = data.iloc[:data_size], data.iloc[data_size:]
    data = data.replace('', np.nan).dropna().reset_index(drop=True)
    if TEST_ON_ANOTHER_DS:
        test_data = test_data.replace('', np.nan).dropna().reset_index(drop=True)
    print(datetime.now() - clean_ts)

    # split to genders
    print('Splitting to genders...')
    male = data.loc[data['gender'] == 'male', ['txt']]
    female = data.loc[data['gender'] == 'female', ['txt']]

    # split to sentences
    print('Splitting to sentences...')
    tokenizer = nltk_data.load('tokenizers/punkt/english.pickle')
    male = male.apply(lambda row: sentecize(row['txt']), axis=1).explode('txt').apply(lambda row: row.strip()).replace('', np.nan).dropna().to_frame('txt').copy().reset_index(drop=True)
    female = female.apply(lambda row: sentecize(row['txt']), axis=1).explode('txt').apply(lambda row: row.strip()).replace('', np.nan).dropna().to_frame('txt').copy().reset_index(drop=True)
    if TEST_ON_ANOTHER_DS:
        test_data['txt'] = test_data.apply(lambda row: sentecize(row['txt']), axis=1)
        test_data['gender'] = test_data.apply(lambda row: [row['gender']] * len(row['txt']), axis=1)
        assert all(test_data['gender'].apply(lambda val: all(v is not None for v in val)))
        test_data = test_data.explode(['txt', 'gender'])
        test_data['txt'] = test_data['txt'].apply(lambda row: row.strip()).replace('', np.nan)
        test_data.dropna(inplace=True)
        test_data.reset_index(drop=True, inplace=True)

    # posify
    print('Posifying...', end=' ')
    ts_pos = datetime.now()
    with Pool(cpu_count() // 2) as pool:
        male['pos'] = pool.map(posify, male['txt'])
        female['pos'] = pool.map(posify, female['txt'])
        if TEST_ON_ANOTHER_DS:
            test_data['pos'] = pool.map(posify, test_data['txt'])
    print(datetime.now() - ts_pos)

    # add labels
    print('Adding labels...')
    male['gender'] = [MALE] * len(male)
    female['gender'] = [FEMALE] * len(female)

    # sample randomly and aggregate
    select_n = CLASS_SIZE if CLASS_SIZE and CLASS_SIZE <= min(len(male), len(female)) else min(len(male), len(female))
    print(f'Class size: {select_n}')
    print('Shuffling and aggregating...')
    male = male.sample(n=select_n, random_state=SEED).reset_index(drop=True)
    female = female.sample(n=select_n, random_state=SEED).reset_index(drop=True)
    male = aggregate(male)
    female = aggregate(female)
    if TEST_ON_ANOTHER_DS:
        test_data = aggregate(test_data)
    print(f'Class size (M,F): {len(male)}, {len(female)}')

    # merge and shuffle
    print('Merging and shuffling...')
    data = pd.concat([male, female]).reset_index(drop=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    if TEST_ON_ANOTHER_DS:
        data_size = len(data)
        data = pd.concat((data, test_data), axis=0).reset_index(drop=True)

    # sentiment analysis
    if FEATURES[FeaturesEnum.SENTIMENT_ANALYSIS]:
        sia = SentimentIntensityAnalyzer()
        sentiments = pd.DataFrame(data['txt'].apply(sia.polarity_scores).tolist())
        sentiments.rename(columns={'pos': 'positive_sentiments', 'neg': 'negative_sentiments', 'neu': 'neutral_sentiments'},
                          inplace=True)
        sentiments.drop('compound', axis=1, inplace=True)
        data = pd.concat([data, sentiments], axis=1).reset_index(drop=True)

    # split to train / test
    print('Splitting to train / test...')
    if TEST_ON_ANOTHER_DS:
        data, test = data.iloc[:data_size], data.iloc[data_size:].reset_index(drop=True)
        test_data = None
    else:
        test = data.iloc[int(len(data) * TRAIN_FRAC):].reset_index(drop=True)
    train = data.iloc[:int(len(data) * TRAIN_FRAC)]
    print(f'Train size: {len(train)}')
    print(f'Test size: {len(test)}')
    print('')

    # Features
    ftr_lst = []
    train_ftr_vals_lst = []
    test_ftr_vals_lst = []
    print('Building features...')
    # 1. tfidf on raw txt
    if FEATURES[FeaturesEnum.TF_IDF_TXT]:
        print('TFIDF on raw...')
        vectorizer_txt = TfidfVectorizer(max_features=1500, ngram_range=(1, 3), use_idf=True, tokenizer=word_tokenize)
        train_features_csr = vectorizer_txt.fit_transform(train['txt'])
        test_features_csr = vectorizer_txt.transform(test['txt'])
        ftr_lst.extend(vectorizer_txt.get_feature_names_out())
        train_ftr_vals_lst.append(train_features_csr)
        test_ftr_vals_lst.append(test_features_csr)
    # 1. tfidf on raw txt chars
    if FEATURES[FeaturesEnum.TF_IDF_TXT_CHAR]:
        print('TFIDF on raw (characters)...')
        vectorizer_txt = TfidfVectorizer(max_features=200, ngram_range=(2, 2), use_idf=True, analyzer='char')
        train_features_char_csr = vectorizer_txt.fit_transform(train['txt'])
        test_features_char_csr = vectorizer_txt.transform(test['txt'])
        ftr_lst.extend(vectorizer_txt.get_feature_names_out())
        train_ftr_vals_lst.append(train_features_char_csr)
        test_ftr_vals_lst.append(test_features_char_csr)

    # 2. tfidf on pos
    if FEATURES[FeaturesEnum.TF_IDF_POS]:
        print('TFIDF on pos...')
        vectorizer_pos = TfidfVectorizer(max_features=1500, ngram_range=(3, 3), use_idf=True, tokenizer=word_tokenize)
        train_features_pos_csr = vectorizer_pos.fit_transform(train['pos'])
        test_features_pos_csr = vectorizer_pos.transform(test['pos'])
        ftr_lst.extend(vectorizer_pos.get_feature_names_out())
        train_ftr_vals_lst.append(train_features_pos_csr)
        test_ftr_vals_lst.append(test_features_pos_csr)

    # 3. average sentence length
    if FEATURES[FeaturesEnum.AVG_SENT_LEN]:
        print('Avg len...')
        ftr_lst.extend(['avg len'])
        train_sent_avg = train['txt'].apply(lambda s: len(s.split()) / len(tokenizer.tokenize(s)))
        test_sent_avg = test['txt'].apply(lambda s: len(s.split()) / len(tokenizer.tokenize(s)))
        scaler = MinMaxScaler()
        train_sent_avg = scaler.fit_transform(np.reshape(train_sent_avg.values, (-1, 1)))
        test_sent_avg = scaler.fit_transform(np.reshape(test_sent_avg.values, (-1, 1)))
        train_ftr_vals_lst.append(train_sent_avg)
        test_ftr_vals_lst.append(test_sent_avg)

    # 4. function and custom words
    if FEATURES[FeaturesEnum.FW_WRDS] or FEATURES[FeaturesEnum.CUSTOM_WRDS]:
        custom_wrds_lst = []
        if FEATURES[FeaturesEnum.FW_WRDS]:
            print('FW words...')
            with open(EXTRA_DIR / FUNCTION_WORDS_LST_FILE) as f:
                custom_wrds_lst.update(f.read().split())
        if FEATURES[FeaturesEnum.CUSTOM_WRDS]:
            print('Custom words...')
            custom_wrds_lst.extend(['wife', 'husband', 'my wife', 'my husband', 'gf', 'bf', 'my gf', 'my bf'])
            custom_wrds_lst.extend(['!', "'ll", ',, i', '., i', '., i, ’', 'and, he', 'but, he', 'didn', 'didn, ’'
                                       , 'didn, ’, t', "do, n't", 'don, ’, t', 'he', 'husband', 'i', 'i, was', 'i, ’', 'i, ’, m'
                                       , 'is', 'it', 'it, ,', 'it, ’', 'it, ’, s', 'm', 'me', 'my', 'my, husband', 'my, wife'
                                       , "n't", 'other, people', 's', 'she', 't', 'was', 'when', 'wife', 'you', '’', '’, m'
                                       , '’, s', '’, t', '“', '”', 'wife', 'my, wife', 'my, husband', 'lmao', 'lol', 'omg',
                                    'sex', 'shit'])

        custom_wrds_lst = list(OrderedDict.fromkeys(custom_wrds_lst))
        ngrams = (min([len(wrds.split()) for wrds in custom_wrds_lst]), max([len(wrds.split()) for wrds in custom_wrds_lst]))
        vectorizer_fw = TfidfVectorizer(vocabulary=custom_wrds_lst, ngram_range=ngrams, use_idf=True, tokenizer=word_tokenize)
        train_features_custom_csr = vectorizer_fw.fit_transform(train['txt'])
        test_features_custom_csr = vectorizer_fw.transform(test['txt'])
        ftr_lst.extend(custom_wrds_lst)
        train_ftr_vals_lst.append(train_features_custom_csr)
        test_ftr_vals_lst.append(test_features_custom_csr)

    # 5. sentiment analysis
    if FEATURES[FeaturesEnum.SENTIMENT_ANALYSIS]:
        sentiment_ftrs = sentiments.columns.tolist()
        ftr_lst.extend(sentiment_ftrs)
        train_ftr_vals_lst.append(train[sentiment_ftrs].values)
        test_ftr_vals_lst.append(test[sentiment_ftrs].values)

    # merge features
    print('Merging features...')
    train_features_merged = sp.hstack(train_ftr_vals_lst, format='csr')
    test_features_merged = sp.hstack(test_ftr_vals_lst, format='csr')
    train_ftr_vals_lst, test_ftr_vals_lst = None, None
    df_train = pd.DataFrame(train_features_merged.toarray(), columns=ftr_lst)
    df_test = pd.DataFrame(test_features_merged.toarray(), columns=ftr_lst)
    train_features_merged = df_train.loc[:, ~df_train.columns.duplicated()]
    test_features_merged = df_test.loc[:, ~df_test.columns.duplicated()]
    df_train, df_test = None, None
    ftr_lst = np.array(train_features_merged.columns.tolist())  # update list because we removed duplicates
    print(f'* Total Features: {len(ftr_lst)}')

    # Train & test
    print('Running classifiers...')
    classifiers = [(SVC(random_state=SEED), False),
                   (LogisticRegression(max_iter=1000, random_state=SEED), True),
                   (MultinomialNB(), True),
                   (RandomForestClassifier(random_state=SEED), True),
                   (KNeighborsClassifier(), False)]

    clfs = []
    for clf, run in classifiers:
        if run:
            print(clf.__class__.__name__)
            clf.fit(train_features_merged, train['gender'])
            clfs.append((clf.__class__.__name__, clf))
            score(clf, train_features_merged, test_features_merged, train['gender'], test['gender'], ftr_lst)

    # Ensemble
    voting_clf = VotingClassifier(estimators=[(clf.__class__.__name__, clf) for clf, run in classifiers if run])
    print(voting_clf.__class__.__name__)
    voting_clf.fit(train_features_merged, train['gender'])
    score(voting_clf, train_features_merged, test_features_merged, train['gender'], test['gender'], ftr_lst, True)

    # select k best
    print('Select K Best ANOVA...')
    #  Features are NOT most to least important. Order is RANDOM.
    best_features = select_k_best_2(ftr_lst, train_features_merged, train['gender'], SELECT_K_BEST)
    print(sorted(list(best_features.values())[0]))
    print(datetime.now() - ts)
