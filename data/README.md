# Word Generation

## Eurlex

Downloaded the EURlex data from [link](http://sourceforge.net/projects/mulan/files/datasets/eurlex-eurovoc-descriptors.rar)

Number of Words found in Glove: 3225 words

- Extract files
- Extract 5000 words
- Save it to "5000_EurlexWords.csv"

## Delicious

Downloaded data from [link](http://sourceforge.net/projects/mulan/files/datasets/delicious.rar)

Number of words found in Glove: 438 Words

- Extract files
- Extract 500 words
- Save it to "500_DeliciousWords.csv"

## Generating Glove Embeddings

Download Glvoe embedding from [link](http://nlp.stanford.edu/data/wordvecs/glove.6B.zip)

- Construct the weight matrix
- If words embedding for word is present in Glove use it
- Else use a random matrix in its place
