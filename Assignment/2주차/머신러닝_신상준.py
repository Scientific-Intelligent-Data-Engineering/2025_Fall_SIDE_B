# 파이썬 ≥3.5 필수
import sys
assert sys.version_info >= (3, 5)

# 사이킷런 ≥0.20 필수
import sklearn
assert sklearn.__version__ >= "0.20"

# 기본 모듈 임포트
import numpy as np
import os

import matplotlib.pyplot as plt

# 그림 저장 위치 지정
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("그림 저장:", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# 불필요한 경고를 무시합니다 (사이파이 이슈 #5998 참조)
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

import os
import tarfile
import urllib.request#python 3.0부터는 urllib.request로 사용

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/liganega/handson-ml2/master/notebooks/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

# tar 파일을 가져와서 지정된 폴더에 압축을 풀면 csv 파일 저장됨.
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
fetch_housing_data()

import pandas as pd

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)
housing = load_housing_data()

from zlib import crc32

def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]
housing_with_id = housing.reset_index()   # `index` 열이 추가된 데이터프레임을 반환합니다
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

# StratifiedShuffleSplit을 사용해 훈련/테스트 세트 분리
from sklearn.model_selection import StratifiedShuffleSplit

housing["income_cat"] = pd.cut(housing["median_income"],
                            bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                            labels=[1, 2, 3, 4, 5])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# 훈련 세트의 크기를 5000개로 제한
# .iloc[:5000]을 사용해 처음 5000개의 샘플만 선택합니다.
strat_train_set = strat_train_set.iloc[:5000]

# 이제 이 5000개로 모든 학습 과정을 진행합니다.
# 중간 주택 가격을 빼고 훈련 세트로 지정
housing = strat_train_set.drop("median_house_value", axis=1)
# 중간 주택 가격은 레이블(타깃)으로 활용
housing_labels = strat_train_set["median_house_value"].copy()

# 이후 코드는 그대로 두면 됩니다.
# housing과 housing_labels는 이제 5000개의 샘플만 포함합니다.
# 나머지 데이터 전처리, 모델 훈련, 평가 과정은 이 5000개 데이터로 이루어집니다.

housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]
corr_matrix = housing.select_dtypes(include=[np.number]).corr()

# 중간 주택 가격을 빼고 훈련 세트로 지정
housing = strat_train_set.drop("median_house_value", axis=1)
# 중간 주택 가격은 레이블(타깃)으로 활용
housing_labels = strat_train_set["median_house_value"].copy()

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)
# 다른 방법: housing_num = housing.select_dtypes(include=[np.number])
imputer.fit(housing_num)
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns,index=housing_num.index)

housing_cat = housing[["ocean_proximity"]]
from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

from sklearn.base import BaseEstimator, TransformerMixin

# 열 인덱스
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

from sklearn.compose import ColumnTransformer

# 수치형 특성 이름들의 리스트
num_attribs = list(housing_num)

# 범주형 특성 이름들의 리스트
cat_attribs = ["ocean_proximity"]
from sklearn.base import BaseEstimator, TransformerMixin

# 열 인덱스
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # *args 또는 **kargs 없음
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # 아무것도 하지 않습니다
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.to_numpy())
housing_extra_attribs = pd.DataFrame(
    housing_extra_attribs,
    columns=list(housing.columns)+["rooms_per_household", "population_per_household"],
    index=housing.index)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.feature_selection import SelectFromModel

# 특성 선택을 위한 SVR 모델 (kernel='linear' 필수)
linear_svr = SVR(kernel='linear')

# SelectFromModel 변환기에 linear_svr 모델을 전달합니다.
# 이 모델의 coef_ 속성을 사용하여 중요한 특성을 선택합니다.
feature_selector = SelectFromModel(
    linear_svr,
    threshold=0.01  # 특성 중요도(계수)가 이 값보다 큰 특성만 선택
)

# 기존 num_pipeline에 SelectFromModel을 추가합니다.
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
    ('feature_selector', feature_selector) # <-- 이 줄을 추가합니다.
])

# 전체 파이프라인은 그대로 사용합니다.
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])

# 이제 housing_prepared에는 선택된 특성들만 포함됩니다.
housing_prepared = full_pipeline.fit_transform(housing, housing_labels)

# 축소된 데이터의 형태를 확인
print("전처리된 데이터 형태:", housing_prepared.shape)

# print(housing_prepared.shape)
# print(housing_prepared[0])
# print(housing_prepared[-1])

#모델 선택과 훈련
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
# print("예측:", lin_reg.predict(some_data_prepared))
# print("레이블:", list(some_labels))

from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)

from sklearn.metrics import mean_absolute_error

lin_mae = mean_absolute_error(housing_labels, housing_predictions)

from sklearn.svm import SVR
svm_reg = SVR(kernel="linear")
svm_reg.fit(housing_prepared, housing_labels)
housing_predictions = svm_reg.predict(housing_prepared)
svm_mse = mean_squared_error(housing_labels, housing_predictions)
svm_rmse = np.sqrt(svm_mse)

my_model = Pipeline([
        ("preparation", full_pipeline),
        ("linear", LinearRegression())
    ])
import joblib
joblib.dump(my_model, "my_model.pkl")
my_model_loaded = joblib.load("my_model.pkl")

my_model.fit(housing, housing_labels)

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {
        'kernel':['linear','rbf'],
        'C':np.arange(1,200,10),
        'gamma':np.arange(0.001, 0.01, 0.001)
    }

svr = SVR()
rnd_search = RandomizedSearchCV(svr, param_distributions=param_distribs,
                                n_iter=10, cv=3, scoring='neg_mean_squared_error', random_state=42)
rnd_search.fit(housing_prepared, housing_labels)
negatie_mse = rnd_search.best_score_
best_mse= -negatie_mse
best_rmse = np.sqrt(best_mse)
print("최적의 하이퍼파라미터:", rnd_search.best_estimator_)
print("최고 점수:", rnd_search.best_score_)
print("최고 RMSE:", best_rmse)

#결과
# 전체 최적값
# 최적의 하이퍼파라미터: SVR(C=np.int64(191), gamma=np.float64(0.01), kernel='linear')
# 최고 점수: -4843794082.913418
# 최고 RMSE: 69597.37123565385

# kernel='rbf'로 고정했을 때 최적값
# 최적의 하이퍼파라미터: SVR(C=np.int64(121), gamma=np.float64(0.006))
# 최고 점수: -13004717626.487436
# 최고 RMSE: 114038.22879406465

# 결국에는 linear 커널이 더 좋은 성능을 보여서 실망스러웠다.
# 게다가 RMSE 값도 7만 정도로 사실 제대로 사용할 수 있는 모델은 아니다.