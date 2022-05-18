# import os
import os 

# tf tools
import tensorflow as tf

# image processsing
from tensorflow.keras.preprocessing.image import (load_img,
                                                  img_to_array,
                                                  ImageDataGenerator)
# VGG16 model
from tensorflow.keras.applications.vgg16 import (preprocess_input,
                                                 decode_predictions,
                                                 VGG16)
# cifar10 data - 32x32
from tensorflow.keras.datasets import cifar10

# layers
from tensorflow.keras.layers import (Flatten, 
                                     Dense, 
                                     Dropout, 
                                     BatchNormalization)
# generic model object
from tensorflow.keras.models import Model

# optimizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD

#scikit-learn
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

# for plotting
import numpy as np
import matplotlib.pyplot as plt




# Preparing the data
def prep_data():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    # Normalize data 
    X_train = X_train/255
    X_test = X_test/255
    # Binary labels are much more computationally efficient, as we are converting our labels/strings into numbers/arrays
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.fit_transform(y_test)
    # Initialize label names
    label_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    
    return (X_train, y_train), (X_test, y_test), label_names


# Build model
def build_model():
    # Initialize model
    model = VGG16(include_top = False, # include_top = classification is always the top layer
             pooling = "avg", 
             input_shape = (32, 32, 3))
    for layer in model.layers:
        layer.trainable = False # Trainable = "run through" is set to false, becaause it shpuld NOT run through the layers 
    
    # Add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation = "relu")(flat1)
    output = Dense(10, activation = "softmax")(class1)

    # Define new model
    model = Model(inputs = model.inputs,
              outputs = output)
    
    # Compile model 
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate = 0.01,
    decay_steps = 10000,
    decay_rate = 0.9)
    sgd = SGD(learning_rate = lr_schedule)
    
    model.compile(optimizer = sgd,
              loss = "categorical_crossentropy",
              metrics = ["accuracy"])
    
    return model



# Evaluate model
def evaluate(model, X_test, y_test, label_names):
    predictions = model.predict(X_test, batch_size=32)
    # print classification report
    print(predictions[0])
    report = classification_report(y_test.argmax(axis=1), 
                                   predictions.argmax(axis=1), 
                                   target_names=label_names)
    print("Script succeeded: The following results shows the classification report, which can also be found in the output-folder")
    print(report)
    
    # Save report 
    with open('output/cnn_report.txt', 'w') as my_txt_file:
        my_txt_file.write(report)


# Plotting 
def plot_history(H, epochs):
    plt.style.use("seaborn-colorblind")

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss", linestyle=":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc", linestyle=":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()
    
    # Saving image
    plt.savefig(os.path.join("output", "cnn_plot.png"))
    

    
# Define main function  
def main():
    (X_train, y_train), (X_test, y_test), label_names = prep_data()
    model = build_model()
    H = model.fit(X_train, y_train,
                        validation_data = (X_test, y_test),
                        epochs = 10,
                        batch_size = 128)
    evaluate(model, X_test, y_test, label_names)
    plot_history(H, 10)
    
    print("Script succeeded: The following results shows the classification report, which can also be found in the output-folder")


# Run main() function from terminal only
if __name__ == "__main__":
    main()
    

