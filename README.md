# MTP2020-RankingXML


## Parameters Used 
Most of the parameters are configured as per given in the paper like (hidden_layer size, embedding size, etc.). We also contacted the paper authors regarding optimal hyperparameters such as m(margin), and lambda. All the parameters are mentioned in params.config under heading of dataset as name.

```
For Eurlex
input_size = 5000
output_size = 3993
embedding_size = 100
attention_layer_size = 25
encoder_layer_size = 1000
hidden_layer_size = 100
learning_rate = 1e-2
epoch_num = 1000
batch_size = 256
m = 10
lamda = 1
```

## Model Details

Model details can be found of model/README.md. Three modules (Feature Embedding, Encoder, Decoder) are used. Batch Norm is added between two FC layers. Initialisation is done by __kaiming initialisation__. For textd ataset is a text dataset Glove embeddings are used. The code for that can be found in _construct_weight_matrix.py_ file, otherwise they are initialised randomly.

Following is the snippet of model for Eurlex data.

```
-----------------------------------------------------------------------
             Layer (type)                Input Shape         Param #
=======================================================================
       FeatureEmbedding-1                [256, 5000]               0
              Embedding-2                [256, 5000]         500,000
                 Linear-3           [256, 5000, 100]           2,525
                   ReLU-4            [256, 5000, 25]               0
                 Linear-5            [256, 5000, 25]           2,600
                Sigmoid-6           [256, 5000, 100]               0
                 Linear-7                 [256, 100]          10,100
                Sigmoid-8                 [256, 100]               0
                Encoder-9                [256, 3993]               0
                Linear-10                [256, 3993]       3,195,200
           BatchNorm1d-11                 [256, 800]           1,600
                  ReLU-12                 [256, 800]               0
                Linear-13                 [256, 800]          80,100
               Sigmoid-14                 [256, 100]               0
               Decoder-15                 [256, 100]               0
                Linear-16                 [256, 100]          80,800
           BatchNorm1d-17                 [256, 800]           1,600
                  ReLU-18                 [256, 800]               0
                Linear-19                 [256, 800]       3,198,393
               Sigmoid-20                [256, 3993]               0
=======================================================================
Total params: 7,072,918
Trainable params: 7,072,918
Non-trainable params: 0
-----------------------------------------------------------------------

```

Rest can be found in the params.config file
## Solver Class

It takes model object, loss function, parameters and lambda as input Following functions are defined in Solver class:
### Fit Function
The model is trained on batches the Bow, tfidf and label are passed to the model, it returns x_hidden, y_hidden and y_predicted. For training Bow and tfidf are passed to Embedding module and thus returning x_hidden, labels are passed to Encoder module thus returning y_hidden which in then are passed to Decoder Module. These are pass to loss function, loss function calculate hidden loss between x_hidden and y_hidden, and reconstruction loss between y_predicted and y_original. These are then merged together by a parameter lambda. Then backward is called on this loss following by the optimizer step.

### Predict Function
Although at the time of model training the Decoder is trained over y_hidden but it is assumed that it'll be able to predict the labels on the basis of x_hidden. For prediction Bow and tfidf are passed to embedding module getting x_hidden which are then passed to Decoder module returning the labels for the given test data.

### net.py
 This file has the implementation of Rank Based Auto-Encoder for multi class classification. Over here we define The feature embedding layer, encoder and 
decoder.
#### Implemented model (Euralex)
-----------------------------------------------------------------------

	   Layer (type)                Input Shape         Param #
       FeatureEmbedding-1                [256, 5000]               0
              Embedding-2                [256, 5000]         500,000
                 Linear-3           [256, 5000, 100]           2,525
                   ReLU-4            [256, 5000, 25]               0
                 Linear-5            [256, 5000, 25]           2,600
                Sigmoid-6           [256, 5000, 100]               0
                 Linear-7                 [256, 100]          10,100
                Sigmoid-8                 [256, 100]               0
                Encoder-9                [256, 3993]               0
                Linear-10                [256, 3993]       3,195,200
           BatchNorm1d-11                 [256, 800]           1,600
                  ReLU-12                 [256, 800]               0
                Linear-13                 [256, 800]          80,100
               Sigmoid-14                 [256, 100]               0
               Decoder-15                 [256, 100]               0
                Linear-16                 [256, 100]          80,800
           BatchNorm1d-17                 [256, 800]           1,600
                  ReLU-18                 [256, 800]               0
                Linear-19                 [256, 800]       3,198,393
               Sigmoid-20                 [256, 3993]              0

	Total params: 7,072,918
	Trainable params: 7,072,918
	Non-trainable params: 0
-----------------------------------------------------------------------


### losses.py
The file losses.py has the implementation  of $L(D)$ which is:
![Loss function](Screenshot from 2020-05-11 23-57-55.png)

![L_ae](Screenshot from 2020-05-11 23-58-11.png)

### Instruction to execute the code
		1. Set the dataset and path to appropriate location in params.config
		2. python main.py <no. of epoch>
