import json
import logging
from typing import List, io

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Embedding, Dense, SpatialDropout1D, GRU, Dropout, LSTM
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

MODEL_FILE_CLASSIFICATION = 'tonality-recognise-classification_test.keras'
TOKENIZER_FILE = 'tokenizer.json'
CLASSES = {0: 'very negative',
           1: 'negative',
           2: 'neutral',
           3: 'positive',
           4: 'very positive'}


def load():
    return load_model(MODEL_FILE_CLASSIFICATION)


def save_tokenizer(tokenizer):
    tokenizer_json = tokenizer.to_json()
    with open(TOKENIZER_FILE, 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))


def load_tokenizer():
    with open('tokenizer.json') as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)
        return tokenizer


def prepare_data():
    text_df = pd.read_csv('datasets/stt/sentlex_exp12.txt', header=None, names=['sentence_id', 'text'])
    labels_df = pd.read_csv('datasets/stt/sentiment_labels.txt', sep='|', header=None, names=['sentence_id', 'label'])

    # Merge dataframes on 'sentence_id'
    df = pd.merge(text_df, labels_df, on='sentence_id')

    # Keep only the text and polarity columns
    df = df[['text', 'label']]
    df['text'] = df['text'].astype('str')

    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(
        pd.cut(df['label'], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=False, include_lowest=True, right=True))

    # Split into training and testing sets
    train_df, test_df = train_test_split(df, test_size=0.05)  # random_state=42
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return df, train_df, test_df


def encode_data(all_data, train_df, test_df):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_data['text'])

    x_train = tokenizer.texts_to_sequences(train_df['text'])
    x_test = tokenizer.texts_to_sequences(test_df['text'])

    vocab_size = len(tokenizer.word_index) + 1

    x_train = pad_sequences(x_train)
    x_test = pad_sequences(x_test)

    return x_train, x_test, vocab_size, tokenizer


def create_model(vocab_size):
    model = Sequential()

    # GRU
    model.add(Embedding(input_dim=vocab_size, output_dim=16))
    model.add(SpatialDropout1D(0.2))
    # model.add(GRU(128, return_sequences=True))
    model.add(GRU(64))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(5, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def evaluate(model, x_test, y_test):
    loss, acc = model.evaluate(x_test, y_test)
    logging.info(f'Test  test loss: {loss}; acc: {acc}')


def predict(model, text, tokenizer=None):
    # Create a new Tokenizer and fit it on the provided text
    tokenizer = load_tokenizer() if not tokenizer else tokenizer
    tokenizer.fit_on_texts([text])

    # Use the tokenizer to encode the sequence
    sequences = tokenizer.texts_to_sequences([text])

    # Pad the sequence to the model's input length
    padded_sequence = pad_sequences(sequences, padding="post", maxlen=model.input_shape[1])

    # Make the tonality prediction
    prediction = model.predict(padded_sequence)

    logging.info(f'raw prediction for <{text}>\n{prediction}')

    return prediction


def parse_tonality(prediction: List[List[float]]) -> str:
    predicted_class = tf.argmax(prediction, axis=1).numpy()[0]
    return CLASSES[predicted_class]


def test_model(model, test_x, test_y, tests=10, tokenizer=None, filename='result.txt'):
    for i in range(tests):
        idx = test_x.sample(n=1).axes[0][0]

        test_text = test_x[idx]
        test_label = test_y[idx]

        prediction = predict(model, test_text, tokenizer=tokenizer)
        predicted_ton = parse_tonality(prediction)
        actual_ton = CLASSES[tf.argmax(test_label).numpy()]

        text = f'''Text <{test_text}>:
Predicted tonality raw: {prediction}; actual tonality raw: {test_label}
Predicted tonality {predicted_ton}; actual tonality: {actual_ton}\n'''
        logging.info(text)


def main():
    logging.basicConfig(level='DEBUG')
    all_data, train_df, test_df = prepare_data()
    x_train, x_test, vocab_size, tokenizer = encode_data(all_data, train_df, test_df)

    y_train = to_categorical(train_df['label'])
    y_test = to_categorical(test_df['label'])

    model = create_model(vocab_size)
    model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))
    model.save(MODEL_FILE_CLASSIFICATION)
    save_tokenizer(tokenizer)

    loss, acc = model.evaluate(x_test, y_test)
    logging.info(f'Test loss: {loss}; acc: {acc}')

    model = load()
    test_model(model, test_df['text'], y_test, tokenizer=tokenizer)


def test():
    tokenizer = load_tokenizer()
    logging.basicConfig(level='INFO')
    _, train_df, test_df = prepare_data()

    y_test = to_categorical(test_df['label'])

    model = load()
    test_model(model, test_df['text'], y_test, tokenizer=tokenizer)


def test_tokenizer():
    tokenizer = load_tokenizer()
    print(len(tokenizer.word_index))

if __name__ == '__main__':
    test_tokenizer()
