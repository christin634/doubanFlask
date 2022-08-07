# -*- coding: utf-8 -*-
import pandas as pd
import jieba
import json
from utils import stopwords_list
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def load_comments(cols):
    """
    读取评论
    :param cols:列表，内容为列序号
    :return: 可迭代对象
    """
    chunk = pd.read_csv('./data/comments.csv', usecols=cols, iterator=True)
    return chunk


def load_movies(cols):
    """
    读取电影
    :param cols: 列表，内容为列序号
    :return: DataFrame对象
    """
    movies = pd.read_csv('./data/movies.csv', usecols=cols)
    return movies


def region_distribute():
    """
    高分电影地区分布
    :return:
    """
    movies = load_movies([6, 13])
    tef = movies.sort_values(by='DOUBAN_SCORE', ascending=False)[:250]
    dic = {}
    for i in tef['REGIONS']:
        ls = [s.strip() for s in i.split('/')]
        for item in ls:
            if dic.get(item) is None:
                dic[item] = 1
            else:
                dic[item] += 1
    dic['中国'] = 0
    ls = []
    for key in dic.keys():
        if '中国' in key and key != '中国':
            dic['中国'] += dic[key]
            ls.append(key)
    for i in ls:
        dic.pop(i)
    dic = sorted(dic.items(), key=lambda x: x[1], reverse=True)
    country, count = [], []
    for i in dic:
        country.append(i[0])
        count.append(i[1])
    dic = {'count': count, 'country': country}
    with open('./data/region_distribute.dat', 'wb') as fd:
        pickle.dump(dic, fd)


def get_region_distribute():
    with open('./data/region_distribute.dat', 'rb') as f:
        dic = pickle.load(f)
    return dic


def cn_comic_scores():
    """
    中日动漫均分对比分析
    :return:
    """
    df = load_movies([0, 6, 8, 13, 18])
    df = df.dropna()
    # 选择23年之前且有评分的动画
    df = df[(df['YEAR'] < 2023) & (df['DOUBAN_SCORE'] > 0) & (df['YEAR'] > 0)]
    df_cn = df[(df['REGIONS'].str.contains('中国')) & (df['GENRES'].str.contains('动画'))]
    df_jp = df[(df['REGIONS'].str.contains('日本')) & (df['GENRES'].str.contains('动画'))]
    # 年转化为datetime64
    df_cn.YEAR = pd.to_datetime(df_cn.YEAR, format='%Y')
    # 以10年为频次重新取样，求评分平均值，保留一位小数
    cn = df_cn.set_index('YEAR')['DOUBAN_SCORE'].resample('10AS').mean().round(1)
    # 年份-1，与下面整数年份对齐
    ls = cn.index.to_list()
    counts = cn.values
    ls = [(date - relativedelta(years=1)) for date in ls]
    cn = pd.Series(counts, index=ls)
    # 年转化为datetime64
    df_jp.YEAR = pd.to_datetime(df_jp.YEAR, format='%Y')
    # 以10年为频次重新取样，求评分平均值，保留一位小数
    jp = df_jp.set_index('YEAR')['DOUBAN_SCORE'].resample('10AS').mean().round(1)
    cn.index = cn.index.year
    jp.index = jp.index.year
    # 数据保存到文件中便于后续处理
    cn.to_pickle('./data/cn_comic_score.dat')
    jp.to_pickle('./data/jp_comic_score.dat')


def get_cn_comic_scores():
    """
    从文件中读取数据，并封装
    :return:
    """
    cn = pd.read_pickle('./data/cn_comic_score.dat')
    jp = pd.read_pickle('./data/jp_comic_score.dat')
    # 横坐标
    xAxis = list(set(cn.index.to_list() + jp.index.to_list()))
    xAxis.sort()
    jp = jp.reindex(xAxis, fill_value=0)
    jp = jp.sort_index(ascending=True)
    cn = cn.sort_index(ascending=True)
    dic = {
        'xAxis': xAxis,
        'jp': jp.values.tolist(),
        'cn': cn.values.tolist()
    }
    return dic


def comments_by_time():
    """
    影评数随时间的变化
    :return:
    """
    movies_df = load_movies([0, 14])
    movies_df = movies_df.dropna()
    movies_df.RELEASE_DATE = pd.to_datetime(movies_df.RELEASE_DATE, infer_datetime_format=True)
    movies_df = movies_df[movies_df.RELEASE_DATE < '2022-07-01']  # 选择特定发行区间的电影
    movies_df = movies_df.set_index('MOVIE_ID', drop=True)
    comments_chunk = load_comments([0, 2, 5])
    df = pd.DataFrame(columns=['COMMENT_ID', 'MOVIE_ID', 'COMMENT_TIME'])
    df = df.set_index('MOVIE_ID', drop=True)
    flag = True
    while flag:  # 分批每次读取1000000条评论
        try:
            comments = comments_chunk.get_chunk(1000000)
            comments.COMMENT_TIME = pd.to_datetime(comments.COMMENT_TIME, infer_datetime_format=True)
            comments = comments.set_index('MOVIE_ID', drop=True)
            df = pd.concat([df, comments])  # 简单处理后，评论连接到一个DataFrame中
        except StopIteration:
            flag = False
            print('Iteration is stopped.')
    new_data = movies_df.merge(df, left_index=True, right_index=True)
    # 计算电影评论与发行日天数差
    new_data['DAYS'] = ((new_data.COMMENT_TIME - new_data.RELEASE_DATE) / pd.Timedelta(1, 'D')).fillna(0).astype(int)
    new_data = new_data[new_data.DAYS > 0]  # 筛选出天数大于0的行
    grouped = new_data.groupby('DAYS')['COMMENT_ID'].count()  # 根据天数分组，计算评论人数
    grouped.to_pickle('./data/comments_by_time.dat')


def get_comments_by_time():
    series = pd.read_pickle('./data/comments_by_time.dat')
    lst = series.to_list()[:400]
    comments_time_lst = []
    for i in range(len(lst)):
        comments_time_lst.append([i, lst[i], "评论数量"])
    return {"comments_time_lst": comments_time_lst}


def score_distribute():
    """
    不同类型电影均分与评论人数、电影数间的关系
    :return:
    """
    movies = load_movies([0, 6, 8, 18])
    movies = movies[(movies.YEAR <= 2022) & (movies.DOUBAN_SCORE > 0)]  # 筛选22年以前且评分大于0的电影
    comments_chunk = load_comments([1, 2])
    flag, ls = True, []
    while flag:  # 分批每次处理1000000条评论
        try:
            comments = comments_chunk.get_chunk(1000000)
            comments = comments.dropna()  # 删除含有NAN数据的列
            df_ = comments.groupby('MOVIE_ID').count()
            ls.append(df_)
        except StopIteration:
            flag = False
            print('Iteration is stopped.')
    for i in range(1, len(ls)):  # 调用add方法，合并同时将相应类型评论人数相加
        ls[0] = ls[0].add(ls[i], fill_value=0)
    sumed_df = ls[0]  # 合并后的DataFrame
    merged_data = movies.merge(sumed_df, on='MOVIE_ID')  # 合并两个表
    cmt_person = merged_data.groupby('GENRES')['USER_MD5'].sum()  # 每个类型评论人数
    # 评论人数由大到小排序取前15种类型
    cmt_person = cmt_person.sort_values(ascending=False)[:15]
    genres = cmt_person.index.to_list()  # 获取评论人数最多的15个类型名
    movies = merged_data[merged_data.GENRES.isin(genres)]  # 选取在15个类型中的电影数据
    movies_group = movies.groupby('GENRES')['MOVIE_ID'].count()
    avg_score = merged_data.groupby('GENRES')['DOUBAN_SCORE'].mean().round(1)  # 求各类型电影均分
    data = pd.concat([cmt_person, avg_score, movies_group], axis=1)  # 连接三个表
    data = data.dropna()
    data.USER_MD5 = data.USER_MD5.astype('Int64')
    data.MOVIE_ID = data.MOVIE_ID.astype('Int64')
    data.to_csv('./data/count_cmt_avg_score.csv')  # 保存到文件


def get_score_distribute():
    series = pd.read_csv('./data/count_cmt_avg_score.csv')
    series.sort_values(by=['USER_MD5', "DOUBAN_SCORE"], inplace=True, ascending=False)
    genres = series['GENRES']
    genres = genres.to_list()
    user_md = series['USER_MD5']
    user_md = user_md.to_list()
    douban_score = series['DOUBAN_SCORE']
    douban_score = douban_score.to_list()
    movie_id = series['MOVIE_ID']
    movie_id = movie_id.to_list()
    score_dis_lst = []
    for i in range(0, 15):
        mid_lst = [user_md[i], douban_score[i], movie_id[i], genres[i]]
        score_dis_lst.append(mid_lst)
    return {"score_dis_lst": score_dis_lst}


def genres_by_time():
    """
    不同类型电影数量随时间的变化关系
    :return:
    """
    df = load_movies([0, 8, 18])
    df = df.dropna()  # 删除含有缺失值的相关行列
    df = df[(df['YEAR'] < 2021) & (df['YEAR'] > 1970)]  # 筛选1970-2021年电影
    df.YEAR = pd.to_datetime(df.YEAR, format='%Y')  # YEAR列转化为DateTime64
    genres = df['GENRES'].value_counts()[:18]
    genres = genres.index.to_list()  # 选择18个类型
    df_ = df[df.GENRES.isin(genres)]  # 筛选出在18个类型中的数据
    # 数据按类型分组后，以10年为间隔重新取样，统计每10年不同类型电影数量
    grouped = df_.set_index('YEAR').groupby('GENRES')['MOVIE_ID'].resample('10AS').count()
    grouped.to_csv('./data/genres_by_time.csv')


def get_genres_by_time():
    movies_08 = pd.read_csv('./data/genres_by_time.csv')
    movies_08.YEAR = pd.to_datetime(movies_08.YEAR, format='%Y-%m-%d')
    year = []
    for i in movies_08.YEAR:
        year.append(i.year)
    year = list(set(year))
    year.sort()

    juqing = movies_08[movies_08.GENRES == '剧情'].MOVIE_ID
    juqing = juqing.values.tolist()

    jd = movies_08[movies_08.GENRES == '剧情/动作'].MOVIE_ID
    jd = jd.values.tolist()

    jx = movies_08[movies_08.GENRES == '剧情/喜剧'].MOVIE_ID
    jx = jx.values.tolist()

    jxa = movies_08[movies_08.GENRES == '剧情/喜剧/爱情'].MOVIE_ID
    jxa = jxa.values.tolist()

    jj = movies_08[movies_08.GENRES == '剧情/惊悚'].MOVIE_ID
    jj = jj.values.tolist()

    ja = movies_08[movies_08.GENRES == '剧情/爱情'].MOVIE_ID
    ja = ja.values.tolist()

    jf = movies_08[movies_08.GENRES == '剧情/犯罪'].MOVIE_ID
    jf = jf.values.tolist()

    action = movies_08[movies_08.GENRES == '动作'].MOVIE_ID
    action = action.values.tolist()

    cartoon = movies_08[movies_08.GENRES == '动画'].MOVIE_ID
    cartoon = cartoon.values.tolist()

    comdy = movies_08[movies_08.GENRES == '喜剧'].MOVIE_ID
    comdy = comdy.values.tolist()

    ca = movies_08[movies_08.GENRES == '喜剧/爱情'].MOVIE_ID
    ca = ca.values.tolist()

    kongbu = movies_08[movies_08.GENRES == '恐怖'].MOVIE_ID
    kongbu = kongbu.values.tolist()

    jingsong = movies_08[movies_08.GENRES == '惊悚'].MOVIE_ID
    jingsong = jingsong.values.tolist()

    jk = movies_08[movies_08.GENRES == '惊悚/恐怖'].MOVIE_ID
    jk = jk.values.tolist()

    aiqing = movies_08[movies_08.GENRES == '爱情'].MOVIE_ID
    aiqing = aiqing.values.tolist()

    fanzui = movies_08[movies_08.GENRES == '犯罪'].MOVIE_ID
    fanzui = fanzui.values.tolist()

    kehuan = movies_08[movies_08.GENRES == '科幻'].MOVIE_ID
    kehuan = kehuan.values.tolist()

    # yingyue = movies_08[movies_08.GENRES == '音乐'].MOVIE_ID
    # yingyue = yingyue.values.tolist()

    dic = {'year': year, 'juqing': juqing, 'jd': jd, 'jx': jx, 'jxa': jxa, 'jj': jj, 'ja': ja, 'jf': jf,
           'action': action,
           'cartoon': cartoon, 'comdy': comdy, 'ca': ca, 'kongbu': kongbu, 'jingsong': jingsong, 'jk': jk,
           'aiqing': aiqing, 'fanzui': fanzui, 'kehuan': kehuan, 'yingyue': yingyue}
    return dic


def emotional_analysis():
    """
    对评论进行情感倾向分析
    :return:
    """
    stopwords = stopwords_list('./data/stopwords.txt')  # 读取停用词
    stopwords = list(set(stopwords))  # 对停⽤词表进⾏去重
    chunk = load_comments([3, 6])
    df = pd.DataFrame(columns=['CONTENT', 'RATING', 'MARK'])
    flag = True
    while flag:  # 对评论进行标记,每次处理1000000条
        try:
            data = chunk.get_chunk(1000000)
            data = data.dropna()
            data.loc[data.loc[:, 'RATING'] >= 3, "MARK"] = 1  # 3分及以上-1-好评
            data.loc[data.loc[:, 'RATING'] < 3, 'MARK'] = 0  # 3分以下-0-差评
            df = pd.concat([df, data])
        except StopIteration:
            flag = False
            print('Iteration is stopped.')
    df = df.dropna()
    # 分割数据集，75%用于训练
    x_train, x_test, y_train, y_test = train_test_split(df.CONTENT, df.MARK, test_size=0.25)
    # 对数据集进行特征抽取
    tf = TfidfVectorizer(stop_words=stopwords)
    # 使用TF-IDF向量化方法获取词频矩阵
    v_x_train = tf.fit_transform(x_train)
    v_x_test = tf.transform(x_test)
    mnb = MultinomialNB(alpha=0.1)
    mnb.fit(v_x_train, y_train)  # 拟合
    y_pred = mnb.predict(v_x_test)  # 预测
    print('评论标记类别为:---\n', y_pred)
    print("准确率为：", mnb.score(v_x_test, y_test))
    # 对某部电影评论进行分析
    chunk = load_comments([2, 3, 6])
    df = pd.DataFrame(columns=['CONTENT', 'RATING'])
    flag = True
    while flag:
        try:
            data = chunk.get_chunk(1000000)
            data = data.dropna()
            data = data[data.MOVIE_ID == 25986180]
            data = data.drop(columns=['MOVIE_ID'])
            df = pd.concat([df, data])
        except StopIteration:
            flag = False
            print('Iteration is stopped.')
    df = df[['CONTENT', 'RATING']]
    df = df[df.MOVIE_ID == 25986180]
    # 原始数据
    raw = pd.Series({
        1.0: df.RATING[df.RATING >= 3].count(),
        0.0: df.RATING[df.RATING < 3].count()
    })
    raw.to_json('./data/raw.json')
    v_x_ = tf.transform(df.CONTENT)
    # 预测值
    y_pred = mnb.predict(v_x_)
    rs = pd.Series(y_pred)
    rs = rs.value_counts()
    rs.to_json('./data/predict_result.json')


def get_emotional_analysis():
    """
    文件中读取原始数据和结果数据
    :return: 字典
    """
    raw = pd.read_json('./data/raw.json', typ='series')
    predict = pd.read_json('./data/predict_result.json', typ='series')
    raw.index = raw.index.second
    predict.index = predict.index.second
    raw_ls = [
        {'value': raw[1], 'name': '好评'},
        {'value': raw[0], 'name': '差评'}
    ]
    predict_ls = [
        {'value': predict[1], 'name': '好评'},
        {'value': predict[0], 'name': '差评'}
    ]
    return {'raw': raw_ls, 'predict': predict_ls}


def mins_scores():
    """
    电影时长与评分的关系
    :return:
    """
    movies_ = load_movies([6, 11, 18])
    # 22年以前且评分大于0的电影
    movies = movies_[(movies_.YEAR <= 2022) & (movies_.DOUBAN_SCORE > 0) & (movies_.MINS >= 60) & (movies_.MINS < 600)]
    movies.drop(columns=['YEAR'], inplace=True)
    movies = movies.groupby(['MINS'])['DOUBAN_SCORE'].mean().round(1)
    # series转化 为字典，key-MINS，value-平均分
    dic = movies.to_dict()
    score, time = [], []
    for i in dic:
        time.append(i)
        score.append(dic[i])
    dic = {'time': time, 'score': score}
    with open('./data/mins_scores.dat', 'wb') as fd:
        pickle.dump(dic, fd)


def get_mins_scores():
    with open('./data/mins_scores.dat', 'rb') as f:
        dic = pickle.load(f)
    return dic


def high_score_wordcloud():
    """
    高分电影类型词云
    :return:
    """
    # 读取电影名，豆瓣评分，电影标签
    genre_mov_data = load_movies([1, 8, 6])
    # 提取豆瓣评分大于7.5的部分
    genre_mov_data = genre_mov_data[genre_mov_data.DOUBAN_SCORE > 7.5]
    # 去掉nan数据
    genre_mov_data = genre_mov_data.dropna()
    # 按豆瓣评分由大到小排序
    genre_mov_data = genre_mov_data.sort_values(by=['DOUBAN_SCORE'], ascending=[False])
    # 提取标签
    data1 = genre_mov_data['GENRES']
    # 将数据转为列表
    genres = data1.values.tolist()
    genres_dict = {}
    # 对标签计数
    for n in genres:
        # 分割带有/的标签
        for key in [s.strip() for s in n.split("/")]:
            if genres_dict.get(key) is None:
                genres_dict[key] = 1
            else:
                genres_dict[key] += 1
    # 将标签按出现次数排序
    genres_dict = sorted(genres_dict.items(), key=lambda x: x[1], reverse=True)
    # 将列表转为字典
    genres_dict = dict(genres_dict)
    # 将数据写入文件
    with open("./data/genres_wordcloud.json", "w", encoding="utf-8") as fp:
        json.dump(genres_dict, fp)
    return pd.Series(genres_dict).value_counts().to_dict()


def get_high_score_wordcloud():
    high_score_cloud_lst = []
    with open("./data/genres_wordcloud.json", "r", encoding="utf-8") as fp:
        dict_ = json.load(fp)
        for k, v in dict_.items():
            item = {"name": k, "value": v}
            high_score_cloud_lst.append(item)
    return {"high_score_cloud_lst": high_score_cloud_lst}


def good_actors():
    """
    高分电影参演演员
    :return:
    """
    actor_mov_data = load_movies([1, 3, 6])
    actor_mov_data = actor_mov_data[actor_mov_data.DOUBAN_SCORE > 0]
    actor_mov_data = actor_mov_data.dropna()
    actor_mov_data = actor_mov_data.sort_values(by=['DOUBAN_SCORE'], ascending=[False])
    actor_mov_data = actor_mov_data.head(250)
    data1 = actor_mov_data['ACTORS']
    # 将数据转为列表
    actors = data1.values.tolist()
    act_names_dict = {}
    for n in actors:
        for key in [s.strip() for s in n.split("/")]:
            if act_names_dict.get(key) is None:
                act_names_dict[key] = 1
            else:
                act_names_dict[key] += 1
    act_names_dict = sorted(act_names_dict.items(), key=lambda x: x[1], reverse=True)
    act_names_dict = dict(act_names_dict)
    with open("./data/actor_name.json", "w", encoding="utf-8") as fp:
        json.dump(act_names_dict, fp)
    return pd.Series(actors).value_counts().to_dict()


def get_good_actors():
    actor_name_lst, actor_fre_lst = [], []
    with open("./data/actor_name.json", "r", encoding="utf-8") as fp:
        dict_ = json.load(fp)
        for k, v in dict_.items():
            actor_name_lst.append(k)
            actor_fre_lst.append(v)
    actor_name_lst = actor_name_lst[:20]
    actor_fre_lst = actor_fre_lst[:20]
    actor_name_lst.reverse()
    actor_fre_lst.reverse()
    return {"actor_name_lst": actor_name_lst, "actor_fre_lst": actor_fre_lst}


def good_director():
    """
    高分电影执导导演
    :return:
    """
    # 读取数据
    dir_mov_data = load_movies([1, 5, 6])
    # 按评分排序
    dir_mov_data = dir_mov_data.sort_values(by=['DOUBAN_SCORE'], ascending=[False])
    # 去掉NAN数据
    dir_mov_data = dir_mov_data.dropna()
    # 提取前250位
    dir_mov_data = dir_mov_data.head(250)
    data1 = dir_mov_data['DIRECTORS']
    director_names = data1.values.tolist()
    director_name_dict = {}
    for n in director_names:
        for key in [s.strip() for s in n.split("/")]:
            if director_name_dict.get(key) is None:
                director_name_dict[key] = 1
            else:
                director_name_dict[key] += 1
    director_name_dict = sorted(director_name_dict.items(), key=lambda x: x[1], reverse=True)
    director_name_dict = dict(director_name_dict)
    with open("./data/director_name_wordcloud.json", "w", encoding="utf-8") as fp:
        json.dump(director_name_dict, fp)
    return pd.Series(director_names).value_counts().to_dict()


def get_good_director():
    director_lst = []
    with open("./data/director_name_wordcloud.json", "r", encoding="utf-8") as fp:
        dict_ = json.load(fp)
        for k, v in dict_.items():
            item = {"name": k, "value": v}
            director_lst.append(item)
    return {"director_lst": director_lst}


if __name__ == '__main__':
    emotional_analysis()

