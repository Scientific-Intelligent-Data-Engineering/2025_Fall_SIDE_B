import sys
assert sys.version_info >= (3, 7)
import sklearn
assert sklearn.__version__ >= "1.0.1"
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

#결정트리 훈련
# a. 초승달 데이터셋
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

X_moons, y_moons = make_moons(n_samples=10000, noise=0.4, random_state=42)

# b. 훈련셋과 테스트셋으로 쪼개기
X_train, X_test, y_train, y_test = train_test_split(X_moons, y_moons, test_size=0.2, random_state=42)

# c. 교차검증을 사용하는 그리드 탐색 실행
params = {'max_leaf_nodes': list(range(10, 31))}
tree_clf = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(tree_clf, params, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

results = pd.DataFrame(grid_search.cv_results_)
print(results[['param_max_leaf_nodes', 'mean_test_score']])
print(f"최적의 하이퍼파라미터: {grid_search.best_params_}")

# d. 테스트셋에 대한 정확도 확인
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"테스트셋 정확도: {accuracy:.4f}")

from sklearn.model_selection import ShuffleSplit
from sklearn.base import clone
from scipy.stats import mode
import numpy as np

# a. 무작위로 선택된 100개의 초승달 훈련 샘플로 구성된 훈련셋을 1000개 생성한다.
n_trees = 1000
n_instances = 100
mini_sets = []
rs = ShuffleSplit(n_splits=n_trees, test_size=len(X_train) - n_instances, random_state=42)

for mini_train_index, mini_test_index in rs.split(X_train):
    X_mini_train = X_train[mini_train_index]
    y_mini_train = y_train[mini_train_index]
    mini_sets.append((X_mini_train, y_mini_train))

print("a. 1000개의 미니 훈련셋 생성 완료")

# b. 앞서 찾은 최적의 모델을 각 미니 훈련셋에 대해 훈련한 다음 테스트셋 정확도의 평균 계산
forest = [clone(best_model) for _ in range(n_trees)]
accuracy_scores = []

for tree, (X_mini_train, y_mini_train) in zip(forest, mini_sets):
    tree.fit(X_mini_train, y_mini_train)
    y_pred = tree.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred))

avg_acc = np.mean(accuracy_scores)
print(f"b. 1000개 모델의 테스트셋 정확도 평균: {avg_acc:.4f}")

# c. 1000개의 모델이 가장 많이 예측하는 값을 예측값으로 사용 (Majority Voting)
Y_pred = np.empty([n_trees, len(X_test)], dtype=np.uint8)
for tree_index, tree in enumerate(forest):
    Y_pred[tree_index] = tree.predict(X_test)
y_pred_majority_votes, n_votes = mode(Y_pred, axis=0)
y_pred_majority_votes = y_pred_majority_votes.reshape([-1])
print("c. 다수결 투표(Voting) 예측값 생성 완료")

# d. 이 방식으로 계산된 예측값을 이용하면 정확도가 상승하는지 확인
voting_accuracy = accuracy_score(y_test, y_pred_majority_votes)
print(f"d. 다수결 투표 모델의 정확도: {voting_accuracy:.4f}")