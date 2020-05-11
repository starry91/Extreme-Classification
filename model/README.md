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