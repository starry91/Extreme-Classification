# Model Architecture

All the initialisation is done by __kaiming initialisation__. A batch norm layer is added between two Fully connected layers. For initialisation of Embedding layer. If the dataset is a text dataset Glove embeddings are used. The code for that can be found in _construct_weight_matrix.py_ file, otherwise they are initialised randomly.

## Embedding Module

- The embedding module has an Embedding layer.
- BOW are passed to Embedding layer.
- Then they are multiplied by tfidf scores.
- These are passed to Attention module which selects the important features.
- The result of Attention modlue are multiplied by score obtained by multiplying embeddings with tf idf
- Average pooling is done on the result
- An FC layer to reduce the dimensions to that of hidden layer size

## Encoder

- Two FC layers
- The output dimensions are of hidden layer size.

## Decoder

- Two FC layers
- The output dimensions are that of label dimension.

## Sample model for Eurlex

```
-----------------------------------------------------------------------
             Layer (type)                Input Shape         Param #
=======================================================================
       FeatureEmbedding-1                [256, 5000]               0
              Embedding-2                [256, 5000]         500,000
                 Linear-3           [256, 5000, 100]           2,525
            BatchNorm1d-4            [256, 5000, 25]          10,000
                   ReLU-5            [256, 5000, 25]               0
                 Linear-6            [256, 5000, 25]           2,600
                Sigmoid-7           [256, 5000, 100]               0
                 Linear-8                 [256, 100]          10,100
                Sigmoid-9                 [256, 100]               0
               Encoder-10                [256, 3993]               0
                Linear-11                [256, 3993]       3,195,200
           BatchNorm1d-12                 [256, 800]           1,600
                  ReLU-13                 [256, 800]               0
                Linear-14                 [256, 800]          80,100
               Sigmoid-15                 [256, 100]               0
               Decoder-16                 [256, 100]               0
                Linear-17                 [256, 100]          80,800
           BatchNorm1d-18                 [256, 800]           1,600
               Sigmoid-19                 [256, 800]               0
                Linear-20                 [256, 800]       3,198,393
=======================================================================
Total params: 7,082,918
Trainable params: 7,082,918
Non-trainable params: 0
-----------------------------------------------------------------------


```