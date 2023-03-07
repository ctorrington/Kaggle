"""
Christopher Torrington, Kaggle Titanic Submission, :)
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

"""
NOTES
 - A note on my spelling. I use normaliZation in code and normaliSation, where I remember, in comments & explanations.
 - Integer Categorical Features appear to be of type int32.
 - Numerical Integer Features appear to be of type float32.
"""


def main():
    batch_size = 32

    column_names = ['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']

    # Reasonings for relevant columns:
        # 'PassengerID' should not have determined if they lived or died.
        # 'Name's of the passengers should not have determined if they lived or died.
        # 'Ticket' numbers of the passengers should not have determined if they lived or died.
        # 'Cabin' 77% of values in cabin are Null.
    relevant_column_names = ['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']

    train_dataset = tf.data.experimental.make_csv_dataset(
        'C:/Users/chris/Documents/Kaggle/Competitions/Titanic/train.csv',
        select_columns = relevant_column_names,
        batch_size = batch_size,
        label_name = 'Survived',
        prefetch_buffer_size = batch_size
    )

    print(f"train_dataset:\n{train_dataset}\n")
    # print(f"train_dataset keys:\n{train_dataset.take(1)}\n")

    # [(train_features, label_batch)] = train_dataset.take(1)
    # print(f"Every feature: {train_features}")
    # print(f"label batch: {label_batch}")
    # print(train_features.keys())



    """
    DATA NORMALISATION
    https://www.tensorflow.org/api_docs/python/tf/keras/layers/Normalization
    """

    normalization_layer = tf.keras.layers.Normalization()
    print("\nTESTING HERE\n")
    print(next(train_dataset.as_numpy_iterator()))
    # normalization_layer.adapt(train_dataset)
    print("normalization_layer adapted to dataset.\n")

    # Normalise the numerical columns.

    # Create a layer that applies feature-wise normalisation to numerical features.
    # This layer will shift and scale inputs into a distribution centered around 0 with standard deviation 1.
    def get_normalization_layer(name, dataset):
        # Create a normalisation layer for the feature.
        normalizer = tf.keras.layers.Normalization(axis = None)

        # Prepare a dataset that only yields the feature.
        feature_dataset = dataset.map(lambda x, y: x[name])
        print(f"Dataset reduced to {name}.\n")

        # Learn the statistics of the data.
        normalizer.adapt(feature_dataset)
        print(f"{name} feature normalised.\n")

        # normalized_data = normalizer(feature_dataset)
        # # normalized_data = normalization_layer(feature_dataset)
        # print(f"normalised data:\n{normalized_data}\n")


        return normalizer
        # return normalized_data

    # Example of normalising numeric features.
    [(train_features, label_batch)] = train_dataset.take(1)
    train_features_age_normalized = train_features['Age']
    # Passenger ages before applying the normalising layer.
    print(f"Passenger ages before normalisation: \n{train_features_age_normalized}\n")
    # Passenger ages after applying the normalising layer.
    example_numerical_layer = get_normalization_layer('Age', train_dataset.take(1))
    print(f"Passenger ages after normalistion: \n {example_numerical_layer(train_features_age_normalized)}\n")


    # Normalise the categorical columns.
    
    # Create a layer that maps values from the vocabulary to integer indices, 
    # & multi-hot encode the features using the tf.keras.layers.StringLookup, tf.keras.layers.IntegerLookup, and tf.keras.CategoryEncoding preprocessing layers
    def get_category_encoding_layer(name, dataset, dtype, max_tokens = None):

        # Create a layer that turns strings into integer indices.
        # Apply multi-hot encoding to the indices with output_mode.
        if dtype == 'string':
            index = tf.keras.layers.StringLookup(max_tokens = max_tokens, output_mode = 'multi_hot')

        # Otherwise, create a layer that turns integer values into integer indices.
        else:
            index = tf.keras.layers.IntegerLookup(max_tokens = max_tokens)

        # Prepare a tf.data.Dataset that only yield the feature.
        feature_dataset = dataset.map(lambda x, y: x[name])

        print(f"feature_dataset:\n{list(feature_dataset)}\n")

        # Learn the set of possible values & assign them a fixed integer index.
        index.adapt(feature_dataset)

        print(f"Adapted dataset:\n{list(feature_dataset)}\n")

        # # Encode the integer indices.
        # encoder = tf.keras.layers.CategoryEncoding(num_tokens = index.vocabulary_size())

        # #  The lamba function captures the layer, so you can use them,
        # # or include them in the Keras Functional model layer.
        # return_value = lambda feature: encoder(index(feature))
        # print(f"returnval: {return_value}")
        return index


    # Example of normalising categorical features.
    [(train_features, label_batch)] = train_dataset.take(1)
    train_features_sex_normalized = train_features['Embarked']
    # Passenger sexes before applying the normalising layer.
    print(f"Passenger sexes before normalisation:\n{train_features_sex_normalized}\n")
    # Passenger sexes after appyling the normalising layer.
    example_categorical_layer = get_category_encoding_layer(name = 'Embarked', dataset = train_dataset.take(1), dtype = 'string')
    print(f"Passenger sexes after normalisation:\n{example_categorical_layer(train_features_sex_normalized)}\n")


    # All of the features to be trained in the dataset will be normalised individually & put into a list,
    # from that list, a layer will be created with 'tf.keras.layers.concatenate' to concetenate all the features in the list,
    # & used as the input to a keras functional model.

    categorical_features = ['Sex', 'Embarked']

    
    all_inputs = []
    encoded_features = []   

    # Normalise the numerical featuress
    numerical_features = ['Age', 'Fare']

    for header in numerical_features:
        numerical_column = tf.keras.Input(shape = (1,), name = header)
        normalization_layer = get_normalization_layer(header, train_dataset)
        encoded_numerical_column = normalization_layer(numerical_column)
        all_inputs.append(encoded_numerical_column)
        print(f"{header} appended to all_inputs.")
        encoded_features.append(encoded_numerical_column)
        print(f"{header} appended to encoded_features.")


    # Turn integer categorical values from the dataset into integer indices
    integer_categorical_features = ['Pclass', 'SibSp', 'Parch',]

    for header in integer_categorical_features:
        integer_categorical_coloumn = tf.keras.layers.Input(shape = (1,), namer = header, dtype = 'int32')
        encoding_layer = get_category_encoding_layer(name = header, dataset = train_dataset, dtype = 'int32')
        encoded_integer_categorical_layer = encoding_layer(integer_categorical_coloumn)
        all_inputs.append(integer_categorical_coloumn)
        encoded_features.append(encoded_integer_categorical_layer)


    


if __name__ == "__main__":
    main()