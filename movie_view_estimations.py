import keras.callbacks
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, Input, Model
from keras.layers import concatenate
from keras.utils.vis_utils import plot_model

max_length = 20
max_tokens = 10000

tf.random.set_seed(1234)


def plot_loss(history: keras.callbacks.History) -> None:
    """
    Plot the loss stored in a model fitting history object.
    :param history: model history.
    """
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Percentage Error')
    plt.legend()
    plt.grid(True)
    plt.show()


def build_regressor() -> None:
    """
    Use the top 250 movie ratings data set to build a regressor that predicts the movie rating count.
    """
    # Import dataset
    column_names = ['Id', 'Title', 'Year', 'Rating', 'Rating Count']
    dataset = pd.read_csv('resources/top-250-movie-ratings.csv', skiprows=1, names=column_names)

    view_column_names = ['Title', 'Year', 'Rating', 'Rating Count', 'Views']
    view_dataset = pd.read_csv('resources/movie-views.csv', skiprows=1, names=view_column_names)

    # Convert rating count string to float
    dataset['Rating Count'] = dataset['Rating Count'] \
        .str.replace(',', '') \
        .astype(np.float64)

    # Remove unneeded data
    dataset.pop('Id')

    # Split data into training and test data set
    train_dataset = view_dataset.sample(frac=0.8, random_state=0)
    test_dataset = view_dataset.drop(train_dataset.index)

    print('test_dataset', test_dataset)

    # Split the feature that is to be predicted from the data
    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    train_targets = train_features.pop('Views')
    test_targets = test_features.pop('Views')

    # Distinguish between numeric and text features
    train_features_numeric = train_features.copy()
    train_features_text = train_features_numeric.pop('Title')

    test_features_numeric = test_features.copy()
    test_features_text = test_features_numeric.pop('Title')

    # Normalize the numeric features
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(train_features_numeric))

    # Configure text processing layer
    max_features = 1000
    sequence_length = 50
    embedding_dim = 64

    def custom_standardization(input_data: str) -> str:
        """
        Custom standardization: lowercase the text.
        :param input_data: text.
        """
        return tf.strings.lower(input_data)

    # Vectorization layers
    vectorize_layer = layers.TextVectorization(
        standardize=custom_standardization,
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=sequence_length)

    def vectorize_text(text: str) -> tf.Tensor:
        """
        Convert text to an integer vector.
        :param text: Text to convert.
        :return: Vectorized text.
        """
        text = tf.expand_dims(text, -1)
        return vectorize_layer(text)

    # Adapt the vectorization layer to the training text
    train_text = train_features['Title']
    vectorize_layer.adapt(train_text)

    # As final preprocessing step, apply the TextVectorization layer to the training and test dataset.
    train_features_text = train_features_text.map(vectorize_text)
    test_features_text = test_features_text.map(vectorize_text)

    # Define the model architecture for a mixed data model: two inputs
    input_numeric = Input(name='input_numeric', shape=(train_features_numeric.shape[1],))
    input_text = Input(name='input_text', shape=(sequence_length,))

    # Branch to handle numeric input
    x = normalizer(input_numeric)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(units=1)(x)
    x = Model(name='model_numeric', inputs=input_numeric, outputs=x)

    # Branch to handle text input
    y = layers.Embedding(max_features + 1, embedding_dim)(input_text)
    y = layers.Dropout(0.2)(y)
    y = layers.GlobalAveragePooling1D()(y)
    y = layers.Dropout(0.2)(y)
    y = layers.Dense(1)(y)
    y = Model(name='model_text', inputs=input_text, outputs=y)

    # Combine output of both branches
    combined = concatenate([x.output, y.output])

    # Apply dense layer and then a regression prediction on the combined outputs
    z = layers.Dense(2, activation="relu")(combined)
    z = layers.Dense(1, activation="linear")(z)

    # our model accepts the inputs of the two branches to output a single value
    model = Model(name='view_estimation_model', inputs=[x.input, y.input], outputs=z)

    model.compile(loss=tf.losses.MeanAbsolutePercentageError(),
                  optimizer=tf.optimizers.Adam(learning_rate=0.05))

    plot_model(model, to_file='./images/model_plot.png', show_shapes=True, show_layer_names=True)

    # Training
    train_features_numeric = np.asarray(train_features_numeric)
    train_features_text = np.vstack(train_features_text)
    train_targets = np.asarray(train_targets)

    test_features_numeric = np.asarray(test_features_numeric)
    test_features_text = np.vstack(test_features_text)
    test_targets = np.asarray(test_targets)

    history = model.fit(
        [train_features_numeric, train_features_text],
        train_targets,
        epochs=250,
        validation_split=0.2  # Calculate validation on 20% of the training data
    )

    # Evaluation
    plot_loss(history)

    test_results = model.evaluate([test_features_numeric, test_features_text], test_targets)

    print('Test results:', test_results)


if __name__ == '__main__':
    build_regressor()
