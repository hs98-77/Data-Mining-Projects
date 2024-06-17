import pandas as pd
from tqdm import tqdm
basket = pd.read_csv('Market_Basket.csv', header=None)
#%%
#1
df=pd.DataFrame()
items = dict()
for i in tqdm(range(len(basket))):
    for j in range(20):
        if not pd.isna(basket.iloc[i][j]):
            if basket.iloc[i][j] in items.keys():
                items[basket.iloc[i][j]]+=1
            else:
                items[basket.iloc[i][j]]=1
items = dict(reversed(sorted(items.items(), key=lambda item: item[1])))
for item in tqdm(items):
    temp = list()
    for i in range(len(basket)):
        if item in list(basket.iloc[i]):
            temp.append(True)
        else:
            temp.append(False)
    df[item]=temp
del temp
del i
del j
del item
#%%
import matplotlib.pyplot as plt
fig, ax=plt.subplots()
#ax.set_xlim(-1,50)
plt.setp(ax.get_xticklabels(), fontsize=10, rotation='vertical')
ax.bar(items.keys(), items.values())
fig = plt.gcf()
fig.set_size_inches(20, 11)
fig.savefig('items.png', dpi=100)
#%%
#2
#a
transactions_count = len(basket)
#%%
#b
items_count = len(items.keys())
#%%
#c
from operator import itemgetter
top5 = dict(sorted(items.items(), key = itemgetter(1), reverse = True)[:5])
#%%
#d
blacktea_count = items["black tea"]
#%%
#3
def appendlength(df, col):
    temp=list()
    for i in df[col]:
        temp.append(len(i))
    df['length']=temp
    return df
#%%
from mlxtend.frequent_patterns import apriori
fis003=appendlength(apriori(df, min_support=0.003, use_colnames=True), 'itemsets')
fis03=appendlength(apriori(df, min_support=0.03, use_colnames=True), 'itemsets')
fis3=appendlength(apriori(df, min_support=0.3, use_colnames=True), 'itemsets')
fis003 = fis003[fis003['length']>=2]
fis03 = fis03[fis03['length']>=2]
fis3 = fis3[fis3['length']>=2]
#%%
from mlxtend.frequent_patterns import fpgrowth
fpgrowth_itemsets=fpgrowth(df, min_support=0.05, use_colnames=True)
#%%
#4
from mlxtend.frequent_patterns import association_rules
frequent_itemsets=apriori(df, min_support=0.03, use_colnames=True)
ar_02=association_rules(frequent_itemsets, metric="confidence", min_threshold=0.2)
#%%
ar_035=association_rules(frequent_itemsets, metric="confidence", min_threshold=0.35)
#%%