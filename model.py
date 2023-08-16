import pandas as pd 
import numpy as np
dataaf = pd.read_csv(r"C:\Users\User\Desktop\Artificial intelligence\Machine learning\Deep learning\Image topic classification\New clean data\2CLEAN.csv")

dataaf.dropna(axis=0, inplace=True)

for i in range(len(dataaf["cleantext"])):
    if dataaf.iloc[i,0]=="sport":
        dataaf.iloc[i,0]="other"
    if dataaf.iloc[i,0]=="science":
        dataaf.iloc[i,0]="other"    

# Label encoder to "label" 
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
dataaf["label"] = label_encoder.fit_transform(dataaf["label"])

class_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Class Label Mapping:", class_mapping)


# Shuffle DataFrame beacause it is sorted in a way that can effect on the model
from sklearn.utils import shuffle
dataaf = shuffle(dataaf)


dataaf.dropna(axis=0, inplace=True)

dataaf["label"].value_counts()
dataaf.isna().sum()





        
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.models import Sequential
from keras.regularizers import l2
from keras.utils import to_categorical

# to make sure there is no float in data
for i in range (len(dataaf["cleantext"])):
    dataaf.iloc[i,1]=str(dataaf.iloc[i,1])


x = dataaf["cleantext"]
y = dataaf["label"].values # Convert "Sentiment" to a NumPy array
tokenizer = Tokenizer()
tokenizer.fit_on_texts(x)
sequences = tokenizer.texts_to_sequences(x)

word_index = tokenizer.word_index
vocab_size = len(word_index) + 1
max_sequence_length = 20 #to control the length of the sequences
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, y, test_size=0.2, random_state=42)


num_classes = len(np.unique(y))# Number of classes

embedding_dim = 6  
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length))
model.add(LSTM(units=60, kernel_regularizer=l2(0.02)))
model.add(Dropout(0.2))
model.add(Dense(units=num_classes, activation='softmax', kernel_regularizer=l2(0.01)))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

y_train_enc = to_categorical(y_train, num_classes=num_classes)
y_test_enc = to_categorical(y_test, num_classes=num_classes)

# Convert data types
X_train = np.asarray(X_train).astype(np.float32)
y_train_enc = np.asarray(y_train_enc).astype(np.int32)
X_test = np.asarray(X_test).astype(np.float32)
y_test_enc = np.asarray(y_test_enc).astype(np.int32)


model.fit(X_train, y_train_enc, batch_size=25, epochs=5)
accuracy = model.evaluate(X_test, y_test_enc)[1]
print("Accuracy:", accuracy)


# saving the model
model.save(r'C:\Users\User\Desktop\Artificial intelligence\Machine learning\Deep learning\Image topic classification\model\img2text_classi_model.h5')

import pickle

file_path = r'C:\Users\User\Desktop\Artificial intelligence\Machine learning\Deep learning\Image topic classification\model/tokenizer.pickle'
with open(file_path, 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
