"""
This is an example script to generate the outcome variable given the input dataset.

This script should be modified to prepare your own submission that predicts 
the outcome for the benchmark challenge by changing the clean_df and predict_outcomes function.

The predict_outcomes function takes a Pandas data frame. The return value must
be a data frame with two columns: nomem_encr and outcome. The nomem_encr column
should contain the nomem_encr column from the input data frame. The outcome
column should contain the predicted outcome for each nomem_encr. The outcome
should be 0 (no child) or 1 (having a child).

clean_df should be used to clean (preprocess) the data.

run.py can be used to test your submission.
"""

# List your libraries and modules here. Don't forget to update environment.yml!
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
#to be added in environment
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.layers import TextVectorization
import re
import string
import joblib
import pickle
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class CustomLayer(keras.layers.Layer):
    def __init__(self, arg1, arg2):
        super().__init__()
        self.arg1 = arg1
        self.arg2 = arg2
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "arg1": self.arg1,
            "arg2": self.arg2,
        })
        return config

    
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, mask=None):
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
        attention_output = self.attention(
            query=inputs, value=inputs, key=inputs, attention_mask=padding_mask
        )
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)


class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [layers.Dense(latent_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
            padding_mask = tf.minimum(padding_mask, causal_mask)

        attention_output_1 = self.attention_1(
            query=inputs, value=inputs, key=inputs, attention_mask=causal_mask
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)

def clean_df(df, background_df):
    """
    Preprocess the input dataframe to feed the model.
    # If no cleaning is done (e.g. if all the cleaning is done in a pipeline) leave only the "return df" command

    Parameters:
    df (pd.DataFrame): The input dataframe containing the raw data (e.g., from PreFer_train_data.csv or PreFer_fake_data.csv).
    background (pd.DataFrame): Optional input dataframe containing background data (e.g., from PreFer_train_background_data.csv or PreFer_fake_background_data.csv).

    Returns:
    pd.DataFrame: The cleaned dataframe with only the necessary columns and processed variables.
    """
    
    df_filt=background_df.copy()
    cols_to_cat = ['brutoink_f', 'brutohh_f']
    for column in cols_to_cat:
     df_filt[column + '_cat'] = 'Q0'  
     df_filt.loc[df[column] > 0, column + '_cat'] = pd.NA  
     df_filt.loc[df[column].isna(), column + '_cat'] = pd.NA

     non_zero_non_na_mask = (df_filt[column] > 0) & (~df_filt[column].isna())
     if non_zero_non_na_mask.any(): 
        quantiles = df_filt.loc[non_zero_non_na_mask, column].quantile([0.0, 0.25, 0.5, 0.75, 0.95, 1]).tolist()
        labels = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']  
        df_filt.loc[non_zero_non_na_mask, column + '_cat'] = pd.cut(df_filt.loc[non_zero_non_na_mask, column], 
                                                                     bins=quantiles, labels=labels, include_lowest=True)

    #print(df_filt['brutoink_f_cat'].value_counts(dropna=False))
    #print(df_filt['brutohh_f_cat'].value_counts(dropna=False))

    cols_to_drop = ['brutoink_f', 'brutohh_f']
    df_filt.drop(columns=cols_to_drop, axis=1, inplace=True)

    df_filt["year"] = df_filt['wave'].astype(str).str[0:4].astype(int)
    df_filt["month"] = df_filt['wave'].astype(str).str[4:].astype(int)
    #df_filt["age"] = (df_filt["year"]-df_filt["birthyear_imp"]).astype(int)     
    #df_filt = df_filt[df_filt["age"]>18]
    # into categories
    # rest were already grouped into categories, just converting the type
 
    columns_to_convert = ["wave", "aantalki", "partner", "burgstat",
                          "woning", "belbezig", "oplzon", "oplmet", "sted",
                          "birthyear_imp", "gender_imp", "migration_background_imp",
                          "brutoink_f_cat", "brutohh_f_cat","year","month"] #"age",

    for column in columns_to_convert:
     df_filt[column] = df_filt[column].astype('category')


    return df_filt


def predict_outcomes(df, background_df, model_path=''):
    """Generate predictions using the saved model and the input dataframe.

    The predict_outcomes function accepts a Pandas DataFrame as an argument
    and returns a new DataFrame with two columns: nomem_encr and
    prediction. The nomem_encr column in the new DataFrame replicates the
    corresponding column from the input DataFrame. The prediction
    column contains predictions for each corresponding nomem_encr. Each
    prediction is represented as a binary value: '0' indicates that the
    individual did not have a child during 2021-2023, while '1' implies that
    they did.

    Parameters:
    df (pd.DataFrame): The input dataframe for which predictions are to be made.
    background_df (pd.DataFrame): The background dataframe for which predictions are to be made.
    model_path (str): The path to the saved model file (which is the output of training.py).

    Returns:
    pd.DataFrame: A dataframe containing the identifiers and their corresponding predictions.
    """
    '''
    ## This script contains a bare minimum working example
    if "nomem_encr" not in df.columns:
        print("The identifier variable 'nomem_encr' should be in the dataset")

    # Load the model
    model = joblib.load(model_path)

    # Preprocess the fake / holdout data
    df = clean_df(df, background_df)

    # Exclude the variable nomem_encr if this variable is NOT in your model
    vars_without_id = df.columns[df.columns != 'nomem_encr']

    # Generate predictions from model, should be 0 (no child) or 1 (had child)
    predictions = model.predict(df[vars_without_id])

    # Output file should be DataFrame with two columns, nomem_encr and predictions
    df_predict = pd.DataFrame(
        {"nomem_encr": df["nomem_encr"], "prediction": predictions}
    )
    '''
    
    columns_to_drop = ["positie", "lftdcat", "lftdhhh", "aantalhh", "woonvorm", "brutoink", "nettoink", "brutocat", "nettocat", "oplcat", "doetmee", "simpc", "netinc", "nettoink_f", "nettohh_f", "werving", "age_imp"] 
    background_df.drop(columns=columns_to_drop, axis=1, inplace=True)

    df_cleaned = clean_df(background_df,background_df)
    
    sequence_length = 165
    model_name=model_path+'model.h5'
    #model_path+
    with open("/app/train_pairs.pkl", 'rb') as f:
     train_pairs = pickle.load(f)
    #print('dentro predict - len(train_pairs): ',len(train_pairs))
    with open(model_path+'tokens.pkl', 'rb') as f:
     tokens = pickle.load(f)

    batch_size = 1024#64
    vocab_size = len(tokens)+4

    #print('Tokens: ', tokens)
    print('Length of Tokens (vocabulary size): ', vocab_size)
    print('Sequence Length: ', sequence_length)

    #print('recovering vectorizer...')
    train_input_texts = [pair[0] for pair in train_pairs]
    #print('dentro predict - len(train_input_texts): ',len(train_input_texts))
    #print('train_input_texts[0]: ',train_input_texts[0])
    #print('type(train_input_texts[0]): ',type(train_input_texts[0]))
    train_output_texts = [pair[1] for pair in train_pairs]
    #print('dentro predict - len(train_output_texts): ',len(train_output_texts))
    
    input_vectorization_from_disk = pickle.load(open(model_path+"eng_vectorization.pkl", "rb"))
    input_vectorization = TextVectorization.from_config(input_vectorization_from_disk['config'])
    #input_vectorization.adapt(tf.data.Dataset.from_tensor_slices([train_input_texts[0]]))
    input_vectorization.adapt(tf.data.Dataset.from_tensor_slices(train_input_texts))
    input_vectorization.set_weights(input_vectorization_from_disk['weights'])

    output_vectorization_from_disk = pickle.load(open(model_path+"spa_vectorization.pkl", "rb"))
    output_vectorization = TextVectorization.from_config(output_vectorization_from_disk['config'])
    #output_vectorization.adapt(tf.data.Dataset.from_tensor_slices([train_output_texts[0]]))
    output_vectorization.adapt(tf.data.Dataset.from_tensor_slices(train_output_texts))
    output_vectorization.set_weights(output_vectorization_from_disk['weights'])

    embed_dim = 128
    latent_dim = 1024
    num_heads = 8
    text_input = []
    #V2
    #var=['year','month','age','aantalki','partner','burgstat','woning','belbezig','oplzon','oplmet',
    #    'sted','gender_imp','migration_background_imp','brutoink_f_cat','brutohh_f_cat']
    
    #V1 'age',
    var=['year','month','sted','gender_imp','migration_background_imp','brutoink_f_cat','brutohh_f_cat']

    unique_nomem_encr=df_cleaned['nomem_encr'].unique()
    #print('dentro predict - len(unique_nomem_encr): ',len(unique_nomem_encr))
    gk = df_cleaned.groupby('nomem_encr')
    for item in unique_nomem_encr:
     #print('nomem_encr: ',item)
     df_nomem_encr=gk.get_group(item)
     eng=str(item)+' '
     for index, row in df_nomem_encr.iterrows():
      for item in var: 
       if not pd.isnull(row[item]):
        if type(row[item])!=str:   
         eng=eng+item+'_'+str(int(row[item]))+' '
        else:
         eng=eng+item+'_'+str(row[item])+' '  

     #spa='[start] '+str(int(row['new_child']))+' [end]'
     text_input.append(eng)

    nomem_encr_text_input=[int(x[0:x.find(' ')]) for x in text_input]
    #text_input=[[x[x.find(' ')+1:],x] for x in text_input]
    text_input=[x[x.find(' ')+1:] for x in text_input]
    print('len(nomem_encr_text_input): ',len(nomem_encr_text_input))
    #print('nomem_encr_text_input[0]: ',nomem_encr_text_input[0])
    print('len(text_input): ',len(text_input))
    #print('text_input[0]: ',text_input[0])
    encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="encoder_inputs")
    x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(encoder_inputs)
    encoder_outputs = TransformerEncoder(embed_dim, latent_dim, num_heads)(x)
    encoder = keras.Model(encoder_inputs, encoder_outputs)

    decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
    encoded_seq_inputs = keras.Input(shape=(None, embed_dim), name="decoder_state_inputs")
    x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(decoder_inputs)
    x = TransformerDecoder(embed_dim, latent_dim, num_heads)(x, encoded_seq_inputs)
    x = layers.Dropout(0.5)(x)
    decoder_outputs = layers.Dense(vocab_size, activation="softmax")(x)
    decoder = keras.Model([decoder_inputs, encoded_seq_inputs], decoder_outputs)

    decoder_outputs = decoder([decoder_inputs, encoder_outputs])
    transformer = keras.Model(
        [encoder_inputs, decoder_inputs], decoder_outputs, name="transformer"
    )
    transformer.load_weights(model_name)

    spa_vocab = output_vectorization.get_vocabulary()
    spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))
    max_decoded_sentence_length = 1
    
    def decode_sequence(input_sentence):
     tokenized_input_sentence = input_vectorization([input_sentence])
     decoded_sentence = "[start]"
     for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = output_vectorization([decoded_sentence])[:, :-1]
        predictions = transformer([tokenized_input_sentence, tokenized_target_sentence])

        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = spa_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token

        if sampled_token == "[end]":
            break
     return decoded_sentence    

    print('Start prediction')
    
    predictions=[]
    pairs=text_input.copy()#train_pairs.copy()
    #print('pairs[0]: ',pairs[0])
    #nomem_encr_pairs=nomem_encr_text_input.copy() #nomem_encr_train_pairs.copy()

    for i in range(len(pairs)):
     input_sentence = pairs[i]
     translated = decode_sequence(input_sentence)
     translated = translated.replace('[start]','')
     translated = translated.replace(' ','')
     #except
     predictions.append(int(translated))

   
    # Output file should be DataFrame with two columns, nomem_encr and predictions
    df_predict = pd.DataFrame(
        {"nomem_encr": nomem_encr_text_input, "prediction": predictions}
    )

    # Return only dataset with predictions and identifier
    return df_predict
