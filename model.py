import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.utils import to_categorical
from sklearn.metrics import f1_score, confusion_matrix,accuracy_score

from spektral.layers import GINConv, GCNConv, TAGConv, GCSConv, GeneralConv, GatedGraphConv


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from spektral.utils.convolution import normalized_adjacency
from sklearn import preprocessing
import random
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

#Model Training Functions
def encode_label(labels):
    label_encoder = preprocessing.LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    labels = to_categorical(labels)
    return labels, label_encoder.classes_

def TAGConvModel(A,F,N,channels,dropout,l2_reg,learning_rate):
  A = TAGConv.preprocess(A).astype('f4')
  X_in = Input(shape=(F, ))
  fltr_in = Input((N, ), sparse=True)

  dropout_1 = Dropout(dropout)(X_in)
  graph_conv_1 = TAGConv(channels,
                         aggregate='mean',
                         K = 6,
                         kernel_regularizer=l2(l2_reg),
                         use_bias=False,
                         )([dropout_1, fltr_in])

  dropout_2 = Dropout(dropout)(graph_conv_1)
  graph_conv_2 = TAGConv(int((channels)/2),
                         aggregate='mean',
                         K = 6,
                         kernel_regularizer=l2(l2_reg),
                         use_bias=False
                         )([dropout_2, fltr_in])

  dropout_3 = Dropout(dropout)(graph_conv_2)
  graph_conv_3 = TAGConv(2,
                         activation='softmax',use_bias=False)([dropout_3, fltr_in])


  model = Model(inputs=[X_in, fltr_in], outputs=graph_conv_3)
  optimizer = Adam(learning_rate=learning_rate)
  model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              weighted_metrics=['acc'])
  model.summary()
  return model

def GCSConvModel(A,F,N,channels,dropout,l2_reg,learning_rate):
  A = GCSConv.preprocess(A).astype('f4')
  X_in = Input(shape=(F, ))
  fltr_in = Input((N, ), sparse=True)

  dropout_1 = Dropout(dropout)(X_in)
  graph_conv_1 = GCSConv(channels,
                         activation='tanh',
                         kernel_regularizer=l2(l2_reg),
                         use_bias=False,
                         )([dropout_1, fltr_in])

  dropout_2 = Dropout(dropout)(graph_conv_1)
  graph_conv_2 = GCSConv(int((channels)/2),
                         activation='tanh',
                         kernel_regularizer=l2(l2_reg),
                         use_bias=False
                         )([dropout_2, fltr_in])

  dropout_3 = Dropout(dropout)(graph_conv_2)
  graph_conv_3 = GCSConv(2,
                         activation='softmax',use_bias=False)([dropout_3, fltr_in])
                    
  model = Model(inputs=[X_in, fltr_in], outputs=graph_conv_3)
  optimizer = Adam(learning_rate=learning_rate)
  model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              weighted_metrics=['acc'])
  model.summary()
  return model

def GCNConvModel(A,F,N,channels,dropout,l2_reg,learning_rate):
  A = GCNConv.preprocess(A).astype('f4')
  X_in = Input(shape=(F, ))
  fltr_in = Input((N, ), sparse=True)

  dropout_1 = Dropout(dropout)(X_in)
  graph_conv_1 = GCNConv(channels,
                         kernel_regularizer=l2(l2_reg),
                         use_bias=False,
                         )([dropout_1, fltr_in])

  dropout_2 = Dropout(dropout)(graph_conv_1)
  graph_conv_2 = GCNConv(int((channels)/2),
                         kernel_regularizer=l2(l2_reg),
                         use_bias=False
                         )([dropout_2, fltr_in])

  dropout_3 = Dropout(dropout)(graph_conv_2)
  graph_conv_3 = GCNConv(2,
                         activation='softmax',use_bias=False)([dropout_3, fltr_in])
  
  model = Model(inputs=[X_in, fltr_in], outputs=graph_conv_3)
  optimizer = Adam(learning_rate=learning_rate)
  model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              weighted_metrics=['acc'])
  model.summary()
  return model

def GeneralConvModel(A,F,N,channels,dropout,l2_reg,learning_rate):
  A = GeneralConv.preprocess(A).astype('f4')
  X_in = Input(shape=(F, ))
  fltr_in = Input((N, ), sparse=True)

  dropout_1 = Dropout(dropout)(X_in)
  graph_conv_1 = GeneralConv(channels,
                         aggregate='sum',
                         activation='tanh',
                         kernel_regularizer=l2(l2_reg),
                         use_bias=False,
                         )([dropout_1, fltr_in])

  dropout_2 = Dropout(dropout)(graph_conv_1)
  graph_conv_2 = GeneralConv(channels,
                         aggregate='sum',
                         activation='tanh',
                         kernel_regularizer=l2(l2_reg),
                         use_bias=False
                         )([dropout_2, fltr_in])

  dropout_3 = Dropout(dropout)(graph_conv_2)
  graph_conv_3 = GeneralConv(2,
                         activation='softmax',use_bias=False)([dropout_3, fltr_in])

  model = Model(inputs=[X_in, fltr_in], outputs=graph_conv_3)
  optimizer = Adam(learning_rate=learning_rate)
  model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              weighted_metrics=['acc'])
  model.summary()
  return model

def GINConvModel(A,F,N,channels,dropout,l2_reg,learning_rate):
  A = GINConv.preprocess(A).astype('f4')
  X_in = Input(shape=(F, ))
  fltr_in = Input((N, ), sparse=True)

  dropout_1 = Dropout(dropout)(X_in)
  graph_conv_1 = GINConv(channels,
                         activation='tanh',
                         mlp_hidden=[int(channels),int(channels/2)],
                         mlp_activation='sigmoid',
                         kernel_regularizer=l2(l2_reg),
                         use_bias=False,
                         )([dropout_1, fltr_in])

  dropout_2 = Dropout(dropout)(graph_conv_1)
  graph_conv_2 = GINConv(int((channels)/2),
                         activation='tanh',
                         mlp_hidden=[int(channels),int(channels/2)],
                         mlp_activation='sigmoid',
                         kernel_regularizer=l2(l2_reg),
                         use_bias=False
                         )([dropout_2, fltr_in])

  dropout_3 = Dropout(dropout)(graph_conv_2)
  graph_conv_3 = GINConv(2,
                         activation='softmax',use_bias=False)([dropout_3, fltr_in])

  model = Model(inputs=[X_in, fltr_in], outputs=graph_conv_3)
  optimizer = Adam(learning_rate=learning_rate)
  model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              weighted_metrics=['acc'])
  model.summary()
  return model

def GatedGraphConvModel(A,F,N,channels,dropout,l2_reg,learning_rate):
  A = GatedGraphConv.preprocess(A).astype('f4')
  X_in = Input(shape=(F, ))
  fltr_in = Input((N, ), sparse=True)

  dropout_1 = Dropout(dropout)(X_in)
  graph_conv_1 = GatedGraphConv(int(channels/4),
                         n_layers = 2,
                         activation='tanh',
                         kernel_regularizer=l2(l2_reg),
                         use_bias=False,
                         )([dropout_1, fltr_in])

  dropout_2 = Dropout(dropout)(graph_conv_1)
  graph_conv_2 = GatedGraphConv(int(channels/4),
                         n_layers = 2,
                         activation='tanh',
                         kernel_regularizer=l2(l2_reg),
                         use_bias=False
                         )([dropout_2, fltr_in])

  dropout_3 = Dropout(dropout)(graph_conv_2)
  graph_conv_3 = GCNConv(2,
                         activation='softmax',use_bias=False)([dropout_3, fltr_in])
                    
  model = Model(inputs=[X_in, fltr_in], outputs=graph_conv_3)
  optimizer = Adam(learning_rate=learning_rate)
  model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              weighted_metrics=['acc'])
  model.summary()
  return model

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.xlabel('Epoch')
  plt.ylabel('Error')
  plt.legend()
  plt.grid(True)
  plt.show()

def train_model(model,X, A, train_mask, val_mask):
  validation_data = ([X, A],labels_encoded,val_mask)
  A = normalized_adjacency(A)

  history = model.fit([X, A],
          labels_encoded,
          sample_weight=train_mask,
          epochs=epochs,
          batch_size=N,
          validation_data=validation_data,
          shuffle=False,
          callbacks = [EarlyStopping(monitor='val_loss', patience=50),
             ModelCheckpoint(filepath='best_model', monitor='val_loss', save_best_only=True)])

  plot_loss(history)

  y_pred = model.predict([X, A], batch_size=N)
  f1_val = f1_score(np.argmax(labels_encoded,axis=1), np.argmax(y_pred,axis=1),zero_division=1,average='micro')
  confusion_val = confusion_matrix(np.argmax(labels_encoded,axis=1), np.argmax(y_pred,axis=1),labels=classes)
  accuracy_val = accuracy_score(np.argmax(labels_encoded,axis=1), np.argmax(y_pred,axis=1))

  print("F1 Score: ",f1_val)
  print("Confusion Matrix: ",confusion_val)
  print("Accuracy Score: ",accuracy_val)


# Main Code
network_data = pd.read_csv("network_final.csv")
nodes_data = pd.read_csv("nodes_final.csv")

nodes_data.corr()

print(network_data.shape)
print(nodes_data.shape)

network_data.head()
nodes_data.head()

le = preprocessing.LabelEncoder()
le.fit(nodes_data['nick'])
nodes_data['nick'] = le.transform(nodes_data['nick'])
network_data['rater nick'] = le.transform(network_data['rater nick'])
network_data['rated nick'] = le.transform(network_data['rated nick'])

nodes = []
labels = []

for index, row in nodes_data.iterrows():
  labels.append(row['tf_new'])
  nodes.append(row['nick'])

nodes = np.asarray(nodes)
labels = np.asarray(labels)

edge_list=[]
for idx,edge in network_data.iterrows():
    edge_list.append((edge['rater nick'],edge['rated nick'],{'rating': edge['rating'],'createdAt': edge['created at(UTC)']}))

G = nx.Graph()
G.add_nodes_from(nodes)
G.add_edges_from(edge_list)

A = nx.adjacency_matrix(G)
print(len(nodes), len(labels), len(edge_list), A.shape)

nodes_data['degree'] = np.zeros(len(nodes_data))

for (node, val) in G.degree():
  nodes_data['degree'][node] = val

X = []

for index, row in nodes_data.iterrows():
  X.append([row['nick'],row['total rating'],row['number of positive ratings received'],row['number of negative ratings received'],row['number of positive ratings sent'],row['number of negative ratings sent'],row['degree']])

print(X)
X = np.array(X,dtype=np.int64)
N = X.shape[0]
F = X.shape[1]
print('X shape: ', X.shape)

print('\nNumber of nodes (N): ', N)
print('\nNumber of features (F) of each node: ', F)

network_data.head()

indexes = [i for i in range(len(nodes))]
random.shuffle(indexes)

train_perc = 0.8
val_perc = 0.1
test_perc = 0.1

train_size = int(train_perc * len(nodes))
val_size = int(val_perc * len(nodes))
test_size = int(test_perc * len(nodes))

train_idx = indexes[:train_size]
val_idx = indexes[train_size:(train_size + val_size)]
test_idx = indexes[(train_size + val_size):]

train_mask = np.zeros((N,),dtype=bool)
train_mask[train_idx] = True

val_mask = np.zeros((N,),dtype=bool)
val_mask[val_idx] = True

test_mask = np.zeros((N,),dtype=bool)
test_mask[test_idx] = True


print(train_mask)
print(val_mask)
print(test_mask)

labels_encoded, classes = encode_label(labels)

channels = 1024      
dropout = 0.5    
l2_reg = 5e-4 
learning_rate = 0.001
epochs = 200  
es_patience = 10

tag_model = TAGConvModel(A,F,N,channels,dropout,l2_reg,learning_rate)
gcs_model = GCSConvModel(A,F,N,channels,dropout,l2_reg,learning_rate)
gcn_model = GCNConvModel(A,F,N,channels,dropout,l2_reg,learning_rate)
general_model = GeneralConvModel(A,F,N,channels,dropout,l2_reg,learning_rate)
gin_model = GINConvModel(A,F,N,channels,dropout,l2_reg,learning_rate)
gated_model = GatedGraphConvModel(A,F,N,channels,dropout,l2_reg,learning_rate)

train_model(tag_model,X,A,train_mask,val_mask)
train_model(gcs_model,X,A,train_mask,val_mask)
train_model(gcn_model,X,A,train_mask,val_mask)
train_model(general_model,X,A,train_mask,val_mask)
train_model(gin_model,X,A,train_mask,val_mask)
train_model(gated_model,X,A,train_mask,val_mask)