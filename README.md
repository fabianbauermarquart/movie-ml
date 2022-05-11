# 1. Movie categorization

## *Problem statement:*

You are given millions of movies and a list of thousands of movie categories (names only e.g. “Sci Fi Movies”, “Romantic Movies”).
Your task is to assign each movie to at least one of the movie categories.
Each movie has a title, description and poster.

## *Chosen approach:*

First, assume we that we have the following data for each movie:
- Title
- Plot summary
- Category

### Solution

We then use supervised learning and the available data to build an LSTM (long short-term memory) classifier.
The plot summary text is vectorized and mapped to a word-embedding.
The categories are one-hot-encoded, and we use softmax for the LSTM network's output activation function.
The data can then be segmented into training and validation sets.
We use the training set to train the classifier using a categorical cross-entropy loss fucntion.

### Evaluation

During training, the quality (i.e. accuracy) of the LSTM classifier can be observed via its accuracy on the evaluation
set to estimate the amount of overfitting.
Additional testing has to be carried out on an additional test data set taken from a different source such that
we can make sure that our classifier is robust against a distribution shift.
This is the case when the validation and test accuracy are equal or higher than the training accuracy.

### Adding a category

The classifier has to be re-trained. 
This is because the LSTM network architecture has to change as the number of output neurons reflects the number of categories.

### Removing a category

1. Easy approach: the classifier output is a list of scores which express the likelihood of each category.
   If a category is removed, we can instead return the category with the 2nd highest score.
2. Integrated approach: we assume that the category has also been removed from the original data.
   Using this modified data, the classifier is trained from the beginning.

### Adding a movie

Simply resume training of the existing classifier with an added batch of training data.

### Removing a movie

No action required. We only have to re-train the classifier if the movie's data was faulty.


| Pros                                                                                                 | Cons                                                                                 |
|------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| LSTMs are state-of-the-art for a lot of NLP tasks, such as machine translation or sentiment analysis | Categories have to be added to training data manually if not already present         |
| The more data we have, the more accurate the model becomes                                           | Depends on quality of data, if descriptions are too short, there may be inaccuracies |
| No need to hand-craft rules to assign movie categories                                               | Can only assign one category per movie                                               |


### Alternative Solution

The solution mentioned above does not handle multi-label assignment.
To tackle this, we modify the classifier: 
The LSTM network's loss function is replaced by a binary crossentropy function.
The output activation function is replaced by a logistic activation function.
Then, we can define a threshold 0.5 which we use to assign the movie categories:
all categories over a score of 0.5 will be assigned to the queried movie.

| Pros                                                                            | Cons                                                                          |
|---------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| We can assign multiple movie categories using only one classifier               | Correlations between categories are not used for classification               |
| Not required to modify the data such as merging co-occuring categories into one | The choice of threshold is heuristic and can only be evaluated after training |
| Not required to train a separate classifier for each category                   |                                                                               |


# 2. Word count in PySpark

## Requirements

- Docker
- Docker images:
  - Spark
- Python 3.6
- Python libraries:
  - PySpark

## Installation

### Docker and Spark

1. Follow the instructions of [Docker](https://docs.docker.com/get-docker/) to install Docker on your machine.

2. Pull the Apache Spark Docker image (this takes some time to download):
   ```bash
   docker run -v ~/spark/work:/home/jovyan/work -d -p 8888:8888 jupyter/pyspark-notebook
   ```

3. Check if the container with the `jupyter/pyspark-notebook` image is running:
   ```bash
   docker ps
   ```

### PySpark (Miniconda)

We recommend users to use [Conda](https://docs.conda.io/en/latest/) to configure the Python environment.

1. Follow the instructions of [Miniconda Installation](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) to install Miniconda.
2. Clone this repository and cd to it.
   ```bash
   git clone https://github.com/fabianbauermarquart/movie-ml.git && cd movie-ml
   ```
3. Use Conda to create and activate a new Conda environment:
   ```bash
   conda create --name spark python=3.6
   conda activate spark
   ```
4. Install the libraries:
   ```bash
   pip install -r requirements.txt
   ```
   
### Download data 

```bash
wget https://s3.amazonaws.com/products-42matters/test/biographies.list.gz --directory-prefix=./resources
gunzip ./resources/biographies.list.gz
```


### Running the code

```bash
spark-submit word_counts.py -i ./resources/biographies2.list -o ./out/biographies_word_count
```


The results are found in the directory `./out/biographies_word_count`.


# 3. Movie view estimations

Follow the same installation instructions as above. 


### Running the code

```bash
python movie_view_estimations.py
```

### Approach

This task is about mixed data: numeric and text.

The available data are (250 data points):
- Movie title
- Year released
- Review score
- Number of reviews

The data to be inferred is:
- Number of views (very limited amount of data, only 6 data points)

Unfortunately, this means we have to perform an inner join of the two datasets because when performing supervised learning,
missing data is of no help and can even lead to overfitting when trying to interpolate.
Because we need to handle mixed data, we need to combine two models, one standard fully-connected deep neural network (for numeric data)
and one that processes word embeddings with several dropout and one pooling layer (for text data),
to perform regression.

The model architecture is shown below:

![Model architecture](./images/model_plot.png)

As loss function, mean absolute percentage error is chosen.
Training is performed over 200 training epochs.

![Training history](./images/history.png)

The final error value after training is:

- Training: 10.42%
- Validation: 9.08%
- Test: 47.57%

While there doesn't occur any overfitting during training, as seen via the validation data,
the model's error on the test set is 47.57%, which is not as good as hoped for,
but expected to improve when more data points are added to view data set.

This approach's pros and cons are listed below:

| Pros                                                            | Cons                         |
|-----------------------------------------------------------------|------------------------------|
| We can regress mixed data, i.e. text and numbers with one model | Test data error is about 48% |
| The model trains quickly in under a minute                      |                              |
| It performs with only about 9% error on the validation set      |                              |


An alternative here would be to use an LSTM layer for handling the text data.
The pros and cons of an LSTM layer are the following:


| Pros                                                 | Cons                                               |
|------------------------------------------------------|----------------------------------------------------|
| Standard for text classification in machine learning | No performance improvement when text is very short |
| Keeps track of words that came earlier in the text   |                                                    |