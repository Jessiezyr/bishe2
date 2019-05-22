import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import os
ds_path = str(pathlib.Path.cwd()) + "/datasets/imdb/" 
np_data = np.load(ds_path + "imdb.npz")
print("np_data keys: ", list(np_data.keys()))  

imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(path=ds_path + "imdb.npz",num_words=10000)

print("Training entries: {}, labels: {}".format(len(train_data),len(train_labels)))

print("First record: {}".format(train_data[0]))

print("Before len:{} len:{}".format(len(train_data[0]),len(train_data[1])))
word_index = imdb.get_word_index(ds_path + "imdb_word_index.json")
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print("The content of first record: ", decode_review(train_data[0]))


train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"],padding='post',maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,                                                       value=word_index["<PAD>"],padding='post',maxlen=256)

print("After - len: {} len: {}".format(len(train_data[0]),len(train_data[1])))  
print("First record: \n", train_data[0]) 


vocab_size = 10000  
model = keras.Sequential()

model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
model.summary()  

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

x_val = train_data[:10000]
partial_x_train = train_data[10000:]
y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]


history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,  # ÑµÁ·ÖÜÆÚ£šÑµÁ·Ä£ÐÍµüŽúÂÖŽÎ£©
                    batch_size=512,  # ÅúÁ¿ŽóÐ¡£šÃ¿ŽÎÌÝ¶ÈžüÐÂµÄÑù±ŸÊý£©
                    validation_data=(x_val, y_val),  # ÑéÖ€ÊýŸÝ
                    verbose=2  )


results = model.evaluate(test_data, test_labels)
print("Result: {}".format(results))


history_dict = history.history  
print("Keys: {}".format(history_dict.keys()))  
loss = history.history['loss']
validation_loss = history.history['val_loss']
accuracy = history.history['acc']
validation_accuracy = history.history['val_acc']
epochs = range(1, len(accuracy) + 1)
plt.subplot(121)  
plt.plot(epochs, loss, 'bo', label='Training loss')  
plt.plot(epochs, validation_loss, 'b', label='Validation loss')  
plt.title('Training and validation loss')  # ±êÌâ
plt.xlabel('Epochs')  # xÖá±êÇ©
plt.ylabel('Loss')  # yÖá±êÇ©
plt.legend()  # »æÖÆÍŒÀý
plt.subplot(122)  # ŽŽœš×ŒÈ·ÂÊËæÊ±Œä±ä»¯µÄÍŒ
plt.plot(epochs, accuracy, color='red', marker='o', label='Training accuracy')
plt.plot(epochs, validation_accuracy, 'r', linewidth=1, label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
#plt.savefig("./outputs/sample-2-figure.png", dpi=200, format='png')
plt.show()  