from flask import Flask as _Flask
from flask import jsonify
from flask import render_template
from flask.json import JSONEncoder as _JSONEncoder
import pandas as pd
import db


# 重写Flask框架中的JSONEncoder类中的default方法
class JSONEncoder(_JSONEncoder):
    def default(self, o):
        import decimal
        if isinstance(o, decimal.Decimal):
            return float(o)
        if isinstance(o, pd.Series):
            return o.to_dict()
        super(JSONEncoder, self).default(o)


class Flask(_Flask):
    json_encoder = JSONEncoder


app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/index')
def home():
    return render_template("index.html")


@app.route('/get_cn_comic_scores')
def get_cn_comic_scores():
    """获取动漫相关数据"""
    data = db.get_cn_comic_scores()
    xAxis = data.get('xAxis')
    cn = data.get('cn')
    jp = data.get('jp')
    return render_template('cn_comic_scores.html', xAxis=xAxis, cn=cn, jp=jp)


@app.route('/get_emotional_analysis')
def get_emotional_analysis():
    """获取电影评价数据"""
    data = db.get_emotional_analysis()
    return render_template('emotional_analysis.html', raw=data.get('raw'), predict=data.get('predict'))


@app.route('/get_region_distribute')
def get_region_distribute():
    """获取评论词云数据"""
    return render_template('region_distribute.html', dic=db.get_region_distribute())


@app.route('/get_mins_scores')
def get_mins_scores():
    return render_template('mins_scores.html', dic=db.get_mins_scores())


@app.route('/get_genres_by_time')
def get_genres_by_time():
    return render_template('genres_by_time.html', dic=db.get_genres_by_time())


@app.route('/get_good_actors')
def get_good_actors():
    """获取演员数据"""
    actor_dic = db.get_good_actors()
    return render_template('good_actors.html', data=actor_dic)


@app.route('/get_good_directors')
def get_good_directors():
    """获取导演数据"""
    director_dic = db.get_good_director()
    return render_template('good_directors.html', data=director_dic)


@app.route('/get_high_score_wordcloud')
def get_high_score_wordcloud():
    """获取演员数据"""
    high_score_cloud_dic = db.get_high_score_wordcloud()
    return render_template('high_score_wordcloud.html', data=high_score_cloud_dic['high_score_cloud_lst'])


@app.route('/get_comments_by_time')
def get_comments_by_time():
    """获取演员数据"""
    comment_time_dic = db.get_comments_by_time()
    return render_template('comments_by_time.html', data=comment_time_dic)


@app.route('/get_score_distribute')
def get_score_distribute():
    """获取演员数据"""
    score_dis_dic = db.get_score_distribute()
    return render_template('score_distribute.html', data=score_dis_dic)


if __name__ == '__main__':
    app.run()
