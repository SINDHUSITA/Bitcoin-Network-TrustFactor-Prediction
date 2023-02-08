# Bitcoin-Network-TrustFactor-Prediction
Aims to improve OTC Trading Credibility through the use of multiple SOTA Graph Neural Network models, with the trade data available to us in the form of a transaction graph

## Abstract
Blockchain Technology has accelerated the usage of digital currencies and cryptocurrency trading. Blockchain technology protects data from any manipulation or theft of trading data. Bitcoin is the largest cryptocurrency chain built with over 200k transactions per day. Bitcoin Trading via Exchanges takes high transaction fees from the traders due to which OTC (Over The Counter) trading has become popular. OTC Trading provides higher profits. The risk with OTC Trading is counterpart fraud. This paper discusses the solution to improve OTC Trading Credibility through the use of data mining techniques. We aim to perform inductive trust prediction using the trade data available to us in the form of a transaction graph.

## Trust Factor Equation
The given data set is a graph of nodes and edges, we derived node characteristics like total rating, the total number of positive ratings given, the total number of the negative rating given, the total number of positive ratings received the total number of negative ratings received during the feature extraction phase. The graph
neural network also needs node labels (trust factor) to create the supervised node classification model. We try 2 approaches that give us different trust factor results and use the labeling function that provides the best distribution of trust factors between 0 and 1. For labeling the data we create different tagging functions that aggregate all the ratings received and sent.

## Approach
Once we calculate the trust-factor label for each node, we add this to the node characteristics as labels for the node. This trust-factor calculation takes into consideration all the ratings for transactions made by the node and does not consider the long path of transient relations between nodes that have not made any transactions yet. 
Hence, we utilize the computation from Graph Neural Networks to calculate this transient relation between all the nodes.


## Methods
We developed multiple Graph Neural Networks using state-of-art algorithms using the dataset and proposed trust-factor evaluations. We compare each of these models and evaluate them in terms of various metrics like accuracy, f1 score, ROC curve, class distribution, etc. For all the methods, we experimented with various combinations of the activation functions from tanh, relu, and sigmoid. The reported model results are the best results each of the models could give based on the hyper-parameters, number of epochs, number of channels, and activation function.

![image](https://user-images.githubusercontent.com/52288575/217594684-ce800ded-c803-49b2-bffb-000720045837.png)

*Contributors: [sindhusita](https://github.com/sindhusita) & [shriram-illini](https://github.com/shriram-illini)*
