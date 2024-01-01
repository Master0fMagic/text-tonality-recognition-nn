import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Embedding, Dense, SpatialDropout1D, GRU, Dropout, LSTM
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

MODEL_FILE_GRU = 'archive/tonality-recognise-gru.keras'
MODEL_FILE_LSTM = 'archive/tonality-recognise-lstm.keras'
MODEL_FILE_CLASSIFICATION = 'archive/tonality-recognise-classification.keras'
MODEL_FILE = 'archive/tonality-recognise.keras'


def prepare_data():
    text_df = pd.read_csv('datasets/stt/sentlex_exp12.txt', header=None, names=['sentence_id', 'text'])
    labels_df = pd.read_csv('datasets/stt/sentiment_labels.txt', sep='|', header=None, names=['sentence_id', 'label'])

    # Merge dataframes on 'sentence_id'
    df = pd.merge(text_df, labels_df, on='sentence_id')

    # Keep only the text and polarity columns
    df = df[['text', 'label']]
    df['text'] = df['text'].astype('str')
    df['label'] = df['label'].astype('float32')

    # Split into training and testing sets
    train_df, test_df = train_test_split(df, test_size=0.05)  # random_state=42

    return train_df, test_df


def prepare_data_classification():
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

    return train_df, test_df


def encode_data(train_df, test_df):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_df['text'])

    x_train = tokenizer.texts_to_sequences(train_df['text'])
    x_test = tokenizer.texts_to_sequences(test_df['text'])

    vocab_size = len(tokenizer.word_index) + 1

    x_train = pad_sequences(x_train)
    x_test = pad_sequences(x_test)

    return x_train, x_test, vocab_size, tokenizer


def create_model_lstm(vocab_size):
    model = Sequential()

    # LSTM
    model.add(Embedding(input_dim=vocab_size, output_dim=40, input_length=40))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(optimizer='nadam', loss='mae', metrics=['mse', 'accuracy'])
    return model


def create_model_gru_classification(vocab_size):
    model = Sequential()

    # GRU
    model.add(Embedding(input_dim=vocab_size, output_dim=16))
    model.add(SpatialDropout1D(0.2))
    model.add(GRU(32, return_sequences=True))
    model.add(GRU(32))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(5, activation='softmax'))

    model.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def create_model_gru(vocab_size):
    model = Sequential()

    # GRU
    model.add(Embedding(input_dim=vocab_size, output_dim=16))
    model.add(GRU(16, return_sequences=True))
    model.add(Dense(8, activation='relu'))
    model.add(LSTM(8, return_sequences=True))
    model.add(Dense(8, activation='relu'))
    model.add(GRU(8))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(optimizer='nadam', loss='mae', metrics=['mse', 'accuracy'])
    return model


def evaluate(model, x_test, y_test):
    loss, mse, acc = model.evaluate(x_test, y_test)
    print(f'Test  mse: {mse}, test loss: {loss}; acc: {acc}')


def predict(model, text, tokenizer=None):
    # Create a new Tokenizer and fit it on the provided text
    tokenizer = Tokenizer() if not tokenizer else tokenizer
    tokenizer.fit_on_texts([text])

    # Use the tokenizer to encode the sequence
    sequences = tokenizer.texts_to_sequences([text])

    # Pad the sequence to the model's input length
    padded_sequence = pad_sequences(sequences, padding="post", maxlen=model.input_shape[1])

    # Make the tonality prediction
    prediction = model.predict(padded_sequence)

    return prediction[0, 0]


def predict_classification(model, text, tokenizer=None):
    # Create a new Tokenizer and fit it on the provided text
    tokenizer = Tokenizer() if not tokenizer else tokenizer
    tokenizer.fit_on_texts([text])

    # Use the tokenizer to encode the sequence
    sequences = tokenizer.texts_to_sequences([text])

    # Pad the sequence to the model's input length
    padded_sequence = pad_sequences(sequences, padding="post", maxlen=model.input_shape[1])

    # Make the tonality prediction
    prediction = model.predict(padded_sequence)

    return prediction


def parse_tonality(ton: float):
    if 0.0 <= ton <= 0.2:
        return "Very Negative"
    elif 0.2 < ton <= 0.4:
        return "Negative"
    elif 0.4 < ton <= 0.6:
        return "Neutral"
    elif 0.6 < ton <= 0.8:
        return "Positive"
    elif 0.8 < ton <= 1.0:
        return "Very Positive"
    else:
        return "Invalid Tonality Prediction"


def test_model(model, test_x, test_y, tests=10, tokenizer=None, filename='result.txt'):
    for i in range(tests):
        idx = test_x.sample(n=1).axes[0][0]

        test_text = test_x[idx]
        test_label = test_y[idx]

        prediction = predict(model, test_text, tokenizer=tokenizer)

        text = f'''Text <{test_text}>:
Predicted tonality: {prediction}; actual tonality: {test_label}\n'''
        with open(filename, mode='a') as f:
            f.write(text)


def test_model_classification(model, test_x, test_y, tests=10, tokenizer=None, filename='result.txt'):
    for i in range(tests):
        idx = test_x.sample(n=1).axes[0][0]

        test_text = test_x[idx]
        test_label = test_y[idx]

        prediction = predict_classification(model, test_text, tokenizer=tokenizer)

        text = f'''Text <{test_text}>:
Predicted tonality: {prediction}; actual tonality: {test_label}\n'''
        # with open(filename, mode='a') as f:
        #     f.write(text)
        print(text)


def main_gru():
    train_df, test_df = prepare_data()
    x_train, x_test, vocab_size, tokenizer = encode_data(train_df, test_df)

    y_train = train_df['label']
    y_test = test_df['label']

    model = create_model_gru(vocab_size)
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    model.save(MODEL_FILE_GRU)

    evaluate(model, x_test, y_test)

    model = load_model(MODEL_FILE_GRU)
    test_model(model, test_df['text'], y_test)


def main_lstm():
    train_df, test_df = prepare_data()
    x_train, x_test, vocab_size, tokenizer = encode_data(train_df, test_df)

    y_train = train_df['label']
    y_test = test_df['label']

    model = create_model_lstm(vocab_size)
    model.fit(x_train, y_train, epochs=150, validation_data=(x_test, y_test))
    model.save(MODEL_FILE_LSTM)

    evaluate(model, x_test, y_test)

    model = load_model(MODEL_FILE_LSTM)
    test_model(model, test_df['text'], y_test)


def main_classification():
    train_df, test_df = prepare_data_classification()
    x_train, x_test, vocab_size, tokenizer = encode_data(train_df, test_df)

    y_train = to_categorical(train_df['label'])
    y_test = to_categorical(test_df['label'])

    model = create_model_gru_classification(vocab_size)
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    model.save(MODEL_FILE_CLASSIFICATION)

    loss, acc = model.evaluate(x_test, y_test)
    print(f'Test loss: {loss}; acc: {acc}')

    model = load_model(MODEL_FILE_CLASSIFICATION)
    test_model_classification(model, test_df['text'], y_test)


def train_more():
    train_df, test_df = prepare_data()
    x_train, x_test, vocab_size, tokenizer = encode_data(train_df, test_df)

    y_train = train_df['label']
    y_test = test_df['label']

    model = load_model(MODEL_FILE)

    # model.compile(optimizer='nadam', loss='mae', metrics=['mse', 'accuracy'])
    # model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
    # model.save(MODEL_FILE_GRU)

    test_model(model, test_df['text'], y_test)


def train_more_classification():
    train_df, test_df = prepare_data_classification()
    x_train, x_test, vocab_size, tokenizer = encode_data(train_df, test_df)

    y_train = to_categorical(train_df['label'])
    y_test = to_categorical(test_df['label'])

    model = load_model(MODEL_FILE_CLASSIFICATION)
    test_model_classification(model, test_df['text'], y_test)


if __name__ == '__main__':
    train_more_classification()
