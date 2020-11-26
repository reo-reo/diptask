# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import lightgbm as lgb
from flask import Flask, request
from werkzeug.datastructures import FileStorage
import pickle
import csv
import os

from flask import Flask, render_template, request, flash, make_response
from wtforms import Form, FloatField, SubmitField, validators, ValidationError

# App config
DEBUG = True
app = Flask(__name__,static_folder="results")
app.config.from_object(__name__)
app.config['SECRET_KEY'] = 'fda0e618-685c-11e7-bb40-fa163eb65161'
app.config["CACHE_TYPE"] = "no-cache"

@app.route('/') 
def runit():
    
    #if os.path.exists("results/result.csv"):
        #os.remove("results/result.csv")
        
    return render_template('pred.html') 

@app.route('/result',methods = ['GET', 'POST'])
def csv_to_df():
    csv_data = request.files.get('data')
        
    if request.method == 'POST':
        if isinstance(csv_data, FileStorage) and csv_data.content_type == 'text/csv':
            train_x = pd.read_csv("train_x.csv")
            train_y = pd.read_csv("train_y.csv")
            test_x = pd.read_csv(csv_data)
            work_no = test_x["お仕事No."]
            all_nan_cols = ["勤務地　最寄駅3（駅名）", "応募先　名称", "勤務地　最寄駅3（沿線名）", "（派遣先）勤務先写真コメント", "勤務地　最寄駅3（分）", "無期雇用派遣", "未使用.14", "（派遣以外）応募後の流れ", "（派遣先）概要　従業員数", "電話応対なし", "週払い", "固定残業制 残業代 下限", "未使用.11", "フリー項目　内容", "先輩からのメッセージ", "未使用.10", "未使用.8", "経験必須", "固定残業制 残業代に充当する労働時間数 下限", "ブロックコード2", "未使用.4", "未使用.7", "メモ", "ブロックコード3", "WEB面接OK", "17時以降出社OK", "寮・社宅あり", "ネットワーク関連のスキルを活かす", "Wワーク・副業可能", "固定残業制 残業代に充当する労働時間数 上限", "プログラム関連のスキルを活かす", "未使用.15", "未使用.12", "エルダー（50〜）活躍中", "人材紹介", "主婦(ママ)・主夫歓迎", "ブロックコード1", "フリー項目　タイトル", "未使用.1", "ブランクOK", "未使用.20", "募集形態", "勤務地　最寄駅3（駅からの交通手段）", "応募先　最寄駅（沿線名）", "仕事写真（下）　写真1　ファイル名", "未使用.16", "仕事写真（下）　写真3　ファイル名", "オープニングスタッフ", "応募先　所在地　ブロックコード", "応募先　所在地　都道府県", "応募先　最寄駅（駅名）", "外国人活躍中・留学生歓迎", "未使用.17", "未使用.9", "日払い", "未使用", "未使用.18", "未使用.22", "未使用.5", "勤務地　周辺情報", "仕事写真（下）　写真2　ファイル名", "バイク・自転車通勤OK", "仕事写真（下）　写真2　コメント", "未使用.3", "未使用.2", "WEB関連のスキルを活かす", "未使用.6", "給与　経験者給与下限", "学生歓迎", "固定残業制 残業代 上限", "未使用.19", "給与　経験者給与上限", "未使用.21", "待遇・福利厚生", "シニア（60〜）歓迎", "ベンチャー企業", "少人数の職場", "仕事写真（下）　写真3　コメント", "応募先　備考", "応募先　所在地　市区町村", "仕事写真（下）　写真1　コメント", "未使用.13", "応募拠点", "これまでの採用者例", "（派遣先）概要　勤務先名（フリガナ）"]
            train_x = train_x.drop(all_nan_cols, axis=1)
            test_x = test_x.drop(all_nan_cols, axis=1)
            all_same_data_cols = ["勤務地固定", "週1日からOK", "ミドル（40〜）活躍中", "ルーティンワークがメイン", "対象者設定　年齢下限", "動画コメント", "給与/交通費　給与支払区分", "CAD関連のスキルを活かす", "固定残業制", "公開区分", "20代活躍中", "検索対象エリア", "就業形態区分", "30代活躍中", "雇用形態", "Dip JobsリスティングS", "資格取得支援制度あり", "対象者設定　年齢上限", "社会保険制度あり", "動画タイトル", "残業月10時間未満", "履歴書不要", "研修制度あり", "DTP関連のスキルを活かす", "新卒・第二新卒歓迎", "産休育休取得事例あり", "動画ファイル名", "対象者設定　性別"]
            #全て同じデータが格納されているカラムの削除
            train_x = train_x.drop(all_same_data_cols, axis=1)
            test_x = test_x.drop(all_same_data_cols, axis=1)

            #行削除の工程で次元の不一致が発生してしまうのを防ぐため、一度目的変数と説明変数を統合する
            apply_number = train_y["応募数 合計"]
            train_x["応募数 合計"] = apply_number

            #統合による混乱を防ぐため、変数名変更
            train_data = train_x

            #重複データの削除
            train_data = train_data.drop_duplicates()
            test_x = test_x.drop_duplicates()
            train_data = train_data.reset_index()
            test_x = test_x.reset_index()
            train_data = train_data.drop(["index"],axis=1)
            test_x = test_x.drop(["index"],axis=1)         

            #カーディナリティの高い文字列データのカラムの削除
            str_cols = ["勤務地　最寄駅2（駅名）", "勤務地　最寄駅2（沿線名）", "お仕事名", "休日休暇　備考", "期間・時間　勤務時間", "（派遣先）概要　事業内容", "（紹介予定）年収・給与例", "（紹介予定）休日休暇", "派遣会社のうれしい特典", "お仕事のポイント（仕事PR）", "（派遣先）職場の雰囲気", "勤務地　最寄駅1（駅名）", "給与/交通費　備考"]
            train_data = train_data.drop(str_cols, axis=1)
            test_x = test_x.drop(str_cols, axis=1)
            
            #使い物にならなさそうなカラムの削除
            invalid_cols = ["（派遣先）勤務先写真ファイル名","勤務地　最寄駅2（分）",]
            train_data = train_data.drop(invalid_cols, axis=1)
            test_x = test_x.drop(invalid_cols, axis=1)

            #欠損値を平均値により補完
            male_percentage = train_data["（派遣先）配属先部署　男女比　男"]
            male_percentage = male_percentage.fillna(male_percentage.mean())
            train_data["（派遣先）配属先部署　男女比　男"] = male_percentage

            male_percentage = test_x["（派遣先）配属先部署　男女比　男"]
            male_percentage = male_percentage.fillna(male_percentage.mean())
            test_x["（派遣先）配属先部署　男女比　男"] = male_percentage

            people_number = train_data["（派遣先）配属先部署　人数"]
            people_number = people_number.fillna(people_number.mean())
            train_data["（派遣先）配属先部署　人数"] = people_number

            people_number = test_x["（派遣先）配属先部署　人数"]
            people_number = people_number.fillna(test_x.mean())
            test_x["（派遣先）配属先部署　人数"] = people_number

            take_minute1 = train_data["勤務地　最寄駅1（分）"]
            take_minute1 = take_minute1.fillna(take_minute1.mean())
            train_data["勤務地　最寄駅1（分）"] = take_minute1

            take_minute1 = train_data["勤務地　最寄駅1（分）"]
            take_minute1 = take_minute1.fillna(take_minute1.mean())
            train_data["勤務地　最寄駅1（分）"] = take_minute1

            age_ave = train_data["（派遣先）配属先部署　平均年齢"]
            age_ave = age_ave.fillna(age_ave.mean())
            train_data["（派遣先）配属先部署　平均年齢"] = age_ave

            age_ave = test_x["（派遣先）配属先部署　平均年齢"]
            age_ave = age_ave.fillna(age_ave.mean())
            test_x["（派遣先）配属先部署　平均年齢"] = age_ave

            train_data['掲載期間　開始日'] = pd.to_datetime(train_data["掲載期間　開始日"])
            train_data['期間・時間　勤務開始日'] = pd.to_datetime(train_data["期間・時間　勤務開始日"])
            train_data['掲載期間　終了日'] = pd.to_datetime(train_data["掲載期間　終了日"])

            test_x['掲載期間　開始日'] = pd.to_datetime(test_x["掲載期間　開始日"])
            test_x['期間・時間　勤務開始日'] = pd.to_datetime(test_x["期間・時間　勤務開始日"])
            test_x['掲載期間　終了日'] = pd.to_datetime(test_x["掲載期間　終了日"])

            date_cols = ["掲載期間　開始日","期間・時間　勤務開始日","掲載期間　終了日"]
            train_data = train_data.drop(date_cols,axis=1)
            test_x = test_x.drop(date_cols, axis=1)

            #one-hot化するカラム 
            ohe_cols = ["拠点番号", "（紹介予定）入社後の雇用形態", "（紹介予定）雇用形態備考", "勤務地　最寄駅2（駅からの交通手段）", "（紹介予定）入社時期", "期間･時間　備考"]
            train_data = pd.get_dummies(train_data, drop_first=True,columns=ohe_cols, dummy_na=True, prefix='dummy',prefix_sep='')
            test_x = pd.get_dummies(test_x, drop_first=True, columns=ohe_cols, dummy_na=True, prefix='dummy', prefix_sep='')

            y = train_data["応募数 合計"]
            X = train_data.drop(["お仕事No.","応募数 合計"],axis=1)

            not_important_cols = ["（派遣先）配属先部署","給与/交通費　給与上限","勤務地　最寄駅1（沿線名）","（派遣）応募後の流れ","勤務地　備考","（派遣先）概要　勤務先名（漢字）","（紹介予定）待遇・福利厚生","勤務地　備考","扶養控除内","休日休暇(月曜日)","Excelのスキルを活かす","（派遣先）配属先部署　男女比　女","WEB登録OK","派遣形態","給与/交通費　交通費","土日祝のみ勤務","経験者優遇","PCスキル不要","英語以外の語学力を活かす","紹介予定派遣","応募資格","仕事内容"]
            X = X.drop(not_important_cols,axis=1)
            test_x = test_x.drop(not_important_cols, axis=1)

            X = X.drop(X.columns[np.isnan(X).any()], axis=1)
            test_x = test_x.dropna(how='all').dropna(how='all', axis=1)

            #one-hot化によって生じる可能性のあるお互いのDataframeにないカラムの補完
            def fill_missing_columns(df_a, df_b):
                columns_for_b = set(df_a.columns) - set(df_b.columns)
                for column in columns_for_b:
                    df_b[column] = 0
                columns_for_a = set(df_b.columns) - set(df_a.columns)
                for column in columns_for_a:
                    df_a[column] = 0

            fill_missing_columns(X, test_x)
            X.sort_index(axis=1, inplace=True)
            test_x.sort_index(axis=1, inplace=True)


            with open('trained_model.pkl', mode='rb') as f:  # with構文でファイルパスとバイナリ読み込みモードを設定
                model = pickle.load(f)   

            y_pred = model.predict(test_x, num_iteration=model.best_iteration)
            pred_comp = pd.DataFrame({"お仕事No.":test_x["お仕事No."],
                "応募数 合計":y_pred})
            pred_comp.to_csv("results/result.csv",index=False)

        else:
            raise ValueError('ファイル形式が異なります。csvファイルを選択してください。')

        return render_template('result.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

    
    
