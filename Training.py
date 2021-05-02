import os
import tensorflow as tf
import matplotlib.pyplot as plt_tr
from sklearn.model_selection import train_test_split


def eval_model_loss(model_history):
    # summarize history for loss
    f, ax = plt_tr.subplots(figsize=(6, 6))
    plt_tr.plot(model_history.history['loss'])
    plt_tr.plot(model_history.history['val_loss'])
    plt_tr.title('model loss')
    plt_tr.ylabel('loss')
    plt_tr.xlabel('epoch')
    plt_tr.legend(['train', 'test'], loc='upper left')
    plt_tr.savefig(os.path.join('results', 'model_loss.png'))


def eval_model_accuracy(model_history):
    # summarize history for accuracy
    f, ax = plt_tr.subplots(figsize=(6, 6))

    plt_tr.plot(model_history.history['accuracy'])
    plt_tr.plot(model_history.history['val_accuracy'])
    plt_tr.title('model accuracy')
    plt_tr.ylabel('accuracy')
    plt_tr.xlabel('epoch')
    plt_tr.legend(['train', 'test'], loc='upper left')
    plt_tr.savefig(os.path.join('results', 'model_accuracy.png'))


class Training:

    epochs = 0
    steps = 0

    def __init__(self, epochs, steps):
        self.epochs = epochs
        self.steps = steps

    def train_on_generator(self, model, data, labels, save_file):

        generator = tf.keras.preprocessing.image.ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=180,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range=0.1,  # Randomly zoom image
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=True)  # randomly flip images

        generator.fit(data)

        # Split train data to features and labels
        training_data, validation_data, training_labels, validation_labels = train_test_split(data, labels,  test_size=0.1)

        # stop if loss isn't changing much
        earlystopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1)

        # Save the best model during the training
        save = os.path.join('results', save_file)
        checkpointer = tf.keras.callbacks.ModelCheckpoint(save,
                                                          monitor='val_loss',
                                                          verbose=1,
                                                          save_best_only=True,
                                                          save_weights_only=True)
        print( "starting to learn")
        training = model.fit(generator.flow(training_data,training_labels, batch_size=100),
                                       epochs=self.epochs,
                                       validation_data=(validation_data, validation_labels),
                                       callbacks=[earlystopper, checkpointer],
                                       steps_per_epoch=self.steps)

        return training
