import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt


network_data = pd.read_csv("network.csv")
nodes_data = pd.read_csv("nodes.csv")
sentiment_data = pd.read_csv("sentiment.csv")

print(network_data.shape)
print(nodes_data.shape)
print(sentiment_data.shape)

network_data.drop_duplicates(inplace=True)
nodes_data.drop_duplicates(inplace=True)
sentiment_data.drop_duplicates(inplace=True)

print(nodes_data.isna().sum())
print(network_data.isna().sum())
print(nodes_data.fillna(0,inplace=True))

print(network_data.shape)
print(nodes_data.shape)
print(sentiment_data.shape)

count = 0
indexes = []
for i in tqdm(nodes_data.index):
  sent_rows = network_data.loc[network_data['rater nick'] == nodes_data['nick'][i]]
  recv_rows = network_data.loc[network_data['rated nick'] == nodes_data['nick'][i]]

  if len(sent_rows) == 0 and len(recv_rows) == 0:
    count += 1
    indexes.append(i)
  elif nodes_data['number of positive ratings received'][i] == 0 and nodes_data['number of negative ratings received'][i] == 0 and nodes_data['number of positive ratings sent'][i] == 0 and nodes_data['number of negative ratings sent'][i] == 0:
    count += 1
    indexes.append(i)

print(count)
print(indexes)

nodes_data.drop(index=indexes,inplace=True)
print(nodes_data.shape)
print(nodes_data.head())

count = 0
indexes = []
for i in tqdm(network_data.index):
  sent_rows = nodes_data.loc[network_data['rater nick'][i] == nodes_data['nick']]
  recv_rows = nodes_data.loc[network_data['rated nick'][i] == nodes_data['nick']]

  if len(sent_rows) == 0 or len(recv_rows) == 0:
    count += 1
    indexes.append(i)

print(count)
print(indexes)

network_data.drop(index=indexes,inplace=True)
print(network_data.shape)
print(network_data.head())

nodes_data.drop_duplicates(inplace=True)
network_data.drop_duplicates(inplace=True)

network_data.head()
nodes_data.head()

sentiment_data = sentiment_data.set_index('id')
sentiment_data.head()

curr_time = int(time.time())
alpha = 0.3
nodes_data['tf'] = np.zeros(len(nodes_data))

for i in tqdm(nodes_data.index):
  sent_rows = network_data.loc[network_data['rater nick'] == nodes_data['nick'][i]]
  credibility_sum = 0
  if len(sent_rows) > 0:
    for index, row in sent_rows.iterrows():
      credibility_sum += row['rating']
  recv_rows = network_data.loc[network_data['rated nick'] == nodes_data['nick'][i]]
  goodness_sum = nodes_data['total rating'][i]/(nodes_data['number of positive ratings received'][i] + nodes_data['number of negative ratings received'][i])

  avg_cred = 0.1
  if len(sent_rows) > 0:
    avg_cred = credibility_sum / len(sent_rows)  
  tf = avg_cred + goodness_sum
  nodes_data['tf'][i] = tf

r = np.asarray(nodes_data['tf'])
normalized = (r-min(r))/(max(r)-min(r))
print(normalized)

tf_class = []
for val in normalized:
  if val < 0.56:
    tf_class.append(0)
  else:
    tf_class.append(1)

print(tf_class)

sns.histplot(data=tf_class)
plt.show()

nodes_data['tf_new'] = tf_class
print(nodes_data['tf_new'].sum())
print(np.where(normalized < 0))

nodes_data.drop(['id', 'keyid','first rated(UTC)'], axis=1, inplace=True)
network_data.drop(['id','rater total rating','notes'],axis=1,inplace=True)

print(nodes_data.shape)
print(network_data.shape)

nodes_data.drop_duplicates(inplace=True)
network_data.drop_duplicates(inplace=True)

print(nodes_data.shape)
print(network_data.shape)

nodes_data.head()

nodes_data.to_csv('nodes_final.csv',index=False)
network_data.to_csv('network_final.csv',index=False)