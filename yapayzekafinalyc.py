# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np


fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0

test_images = test_images / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)
train_loss, train_acc = model.evaluate(train_images,  train_labels, verbose=2)
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
results=[]
txt = "Accuracy for all train data {acc:.2f}"

results.append(txt.format(acc=train_acc))
txt = "Accuracy for all test data {acc:.2f}"

results.append(txt.format(acc=test_acc))
for i in range(0,10):
    index=0
    sub_train_x=[]
    sub_train_y=[]
    for label in train_labels:
        if label==i:
            sub_train_y.append(label)
            sub_train_x.append(train_images[index])
        index =index+1
    stx=np.array(sub_train_x)
    sty=np.array(sub_train_y)
    sub_loss,sub_acc=model.evaluate(stx,  sty, verbose=2)
    txt = "Accuracy for {label:}  train data {acc:.2f}"
    results.append(txt.format(acc=sub_acc,label=class_names[i]))

for i in range(0,10):
    index=0
    sub_train_x=[]
    sub_train_y=[]
    for label in test_labels:
        if label==i:
            sub_train_y.append(label)
            sub_train_x.append(test_images[index])
        index =index+1
    stx=np.array(sub_train_x)
    sty=np.array(sub_train_y)
    sub_loss,sub_acc=model.evaluate(stx,  sty, verbose=2)
    txt = "Accuracy for {label:}  test data {acc:.2f}"
    results.append(txt.format(acc=sub_acc,label=class_names[i]))

f = open("SONUC.txt", "w")

for result in results:
    f.write(result+str("\n"))
f.close()

