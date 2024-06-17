import pandas as pd
dir = "DM_PR_DATASET"

values = pd.read_csv(dir+"/values.csv")
labels = pd.read_csv(dir+"/labels.csv")
values = values.drop('building_id', axis=1)
#%%
land_condition_encode = list(set(values['land_surface_condition']))
values['land_surface_condition'] = values.land_surface_condition.map(lambda x: land_condition_encode.index(x))
#%%
foundation_type_encode = list(set(values['foundation_type']))
values['foundation_type'] = values.foundation_type.map(lambda x: foundation_type_encode.index(x))
#%%
roof_type_encode = list(set(values['roof_type']))
values['roof_type'] = values.roof_type.map(lambda x: roof_type_encode.index(x))
#%%
ground_floor_type_encode = list(set(values['ground_floor_type']))
values['ground_floor_type'] = values.ground_floor_type.map(lambda x: ground_floor_type_encode.index(x))
#%%
other_floor_type_encode = list(set(values['other_floor_type']))
values['other_floor_type'] = values.other_floor_type.map(lambda x: other_floor_type_encode.index(x))
#%%
position_encode = list(set(values['position']))
values['position'] = values.position.map(lambda x: position_encode.index(x))
#%%
plan_configuration_encode = list(set(values['plan_configuration']))
values['plan_configuration'] = values.plan_configuration.map(lambda x: plan_configuration_encode.index(x))
#%%
legal_ownership_status_encode = list(set(values['legal_ownership_status']))
values['legal_ownership_status'] = values.legal_ownership_status.map(lambda x: legal_ownership_status_encode.index(x))
#%%
features = pd.DataFrame()
#%%
from scipy.stats import binned_statistic
values['geo_level_1_id'] = binned_statistic(values['geo_level_1_id'], values['geo_level_1_id'], bins=3, range=(0, 1))[2]
values['geo_level_2_id'] = binned_statistic(values['geo_level_2_id'], values['geo_level_2_id'], bins=3, range=(0, 1))[2]
values['geo_level_3_id'] = binned_statistic(values['geo_level_3_id'], values['geo_level_3_id'], bins=3, range=(0, 1))[2]

#%%
features['has_superstructure_mud_mortar_stone'] = values['has_superstructure_mud_mortar_stone']
features['has_superstructure_mud_mortar_brick'] = values['has_superstructure_mud_mortar_brick']
features['has_superstructure_rc_non_engineered'] = values['has_superstructure_rc_non_engineered']
features['has_superstructure_rc_engineered'] = values['has_superstructure_rc_engineered']
#%%
import math
values['age'] = values.age.map(lambda x: math.log(x+1))
#%%
from tqdm import tqdm
superstructure = list()
supstr_dict = {'has_superstructure_mud_mortar_stone':0, 'has_superstructure_stone_flag':1, 'has_superstructure_cement_mortar_stone':2, 'has_superstructure_mud_mortar_brick':3, 'has_superstructure_cement_mortar_brick':4, 'has_superstructure_timber':5, 'has_superstructure_bamboo':6, 'has_superstructure_rc_non_engineered':7, 'has_superstructure_rc_engineered':8, 'has_superstructure_other':9, 'has_superstructure_adobe_mud':10}
for i in tqdm(range(len(values))):
  for k in supstr_dict.keys():
    if values.iloc[i][k]:
      superstructure.append(supstr_dict[k])
      break
features['superstructure'] = superstructure
#%%
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neural_network import MLPClassifier
model = MLPClassifier(alpha=3e-5, hidden_layer_sizes=32, learning_rate='adaptive', validation_fraction=0.25, verbose=False, activation='tanh')
sfs = SequentialFeatureSelector(model, n_features_to_select=10)
sfs.fit(values[:1000], labels[:1000]['damage_grade'])
#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(values, labels['damage_grade'], stratify=labels['damage_grade'], test_size=0.2)
#%%
X_train_sfs = sfs.transform(X_train)
X_test_sfs = sfs.transform(X_test)
#%%
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
model = SVC()
pred = model.fit(X_train_sfs[:10000],y_train[:10000]).predict(X_test_sfs)
print(accuracy_score(y_test, pred))
#%%
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
clf = MLPClassifier(alpha=3e-5, hidden_layer_sizes=8, learning_rate='adaptive', validation_fraction=0.25, verbose=True, activation='tanh')
clf.fit(X_train_sfs, y_train)
pred = clf.predict(X_test_sfs)
print(accuracy_score(y_test, pred))
#%%
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
chi2_fs = SelectKBest(chi2, k=10)
cfs_features = chi2_fs.fit_transform(values, labels)
#%%
import seaborn as sb
import matplotlib.pyplot as plt
cor = values.corr()
plt.figure(figsize=(40,30))
sb.heatmap(cor, annot=True)
#%%
from sklearn.feature_selection import VarianceThreshold
vt = VarianceThreshold(threshold=0)
vt.fit(values)
print(vt.get_support())
#%%
import numpy as np
mad = np.sum(np.abs(values-np.mean(values,axis=0)),axis=0)/values.shape[0]
plt.bar(np.arange(values.shape[1]), mad, color='blue')
#%%
clf_acc = dict()
for size in [8, 16, 32, 64]:
    acc = accuracy_score(y_test, MLPClassifier(alpha=3e-5, hidden_layer_sizes=size, learning_rate='adaptive', validation_fraction=0.25, verbose=True, activation='tanh', max_iter=500).fit(X_train_sfs).predict(X_test_sfs))
    clf_acc[size] = acc
del acc
del size
#%%