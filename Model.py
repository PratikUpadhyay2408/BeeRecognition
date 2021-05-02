import tensorflow as tf


class Model:
    # input shape = (img_height, img_width, channels)
    input_shape = (0,0,0)
    # number of labels in the output vector
    output_size = 0

    def __init__(self, input_shape, output_size):
        self.input_shape = input_shape
        self.output_size = output_size

    def cnn_model(self):
        # Build and train CNN model
        cnnmodel = tf.keras.models.Sequential()
        cnnmodel.add(tf.keras.layers.Conv2D(6, kernel_size=3, input_shape=self.input_shape, activation='relu'))
        cnnmodel.add(tf.keras.layers.MaxPool2D(2))
        cnnmodel.add(tf.keras.layers.Conv2D(12, kernel_size=3, activation='relu'))
        cnnmodel.add(tf.keras.layers.Flatten())
        cnnmodel.add(tf.keras.layers.Dense(self.output_size, activation='softmax'))
        cnnmodel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return cnnmodel

    def vgg_model(self):
        # Build and train CNN model
        vgg = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=self.input_shape )

        for layer in vgg.layers[:5]:
            layer.trainable = False

        x = tf.keras.layers.Flatten()(vgg.output)
        x = tf.keras.layers.Dense(4096, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(4096, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        predictions = tf.keras.layers.Dense(self.output_size , activation='softmax')(x)

        # create graph of your new model
        head_model = tf.keras.models.Model(inputs=vgg.input,outputs=predictions)

        # compile the model
        head_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        head_model.summary()

        return head_model
