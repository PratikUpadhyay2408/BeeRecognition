import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt_main
from Data_Processing import Data
from Model import Model
import Training

# GLOBALS
img_width=100
img_height=100
img_channels=3
bee_data_path = r'D:\AI\data\bee_data.csv'
bee_images = r'D:\AI\data\bee_imgs\bee_imgs'
save_file = r'cnn_model.h5'

if __name__ == "__main__":

    data = Data(bee_data_path, bee_images, img_width, img_height, img_channels)
    bees = data.load_data()
    bees['img'] = bees['file'].apply(data.read_img)

    if not os.path.isdir('results'):
        os.mkdir('results')

    data.plot_subspecies(bees)

    # balance dataset on subspecies
    train_bees, test_bees = data.balanced_data_split(bees)

    # split specifically for subspecies recognition and encode one hot
    train_labels = pd.get_dummies(train_bees['subspecies'])
    train_data = np.stack(train_bees['img'])
    test_labels = pd.get_dummies( test_bees['subspecies'])
    test_data = np.stack(test_bees['img'])
    print(test_bees['subspecies'].unique())

    # Create the model and compile it.
    input_size = (img_height,img_width,img_channels)
    output_size = train_labels.columns.size
    ai_model = Model(input_size, output_size)
    cnn_model = ai_model.cnn_model()

    # Initialize training params, train and eval model.
    train_model = Training.Training(epochs=25, steps=80)
    training = train_model.train_on_generator(cnn_model, train_data, train_labels, save_file=save_file)
    Training.eval_model_loss(model_history=training)
    Training.eval_model_accuracy(model_history=training)

    # Evaluate on test data
    cnn_model.load_weights( os.path.join('results', save_file) )
    plt_main.subplots(figsize=(6, 8))
    test_pred = cnn_model.predict(test_data)
    acc_by_subspecies = np.logical_and((test_pred > 0.5), test_labels).sum()/test_labels.sum()
    acc_by_subspecies.plot(kind='bar', title='Subspecies prediction accuracy')
    plt_main.xticks(rotation=45, fontsize=7)
    plt_main.ylabel('Accuracy')
    plt_main.savefig(os.path.join('results', 'test_accuracy.png'))

    # Loss function and accuracy
    test_res = cnn_model.evaluate(test_data, test_labels)
    print('Evaluation: loss function: %s, accuracy:' % test_res[0], test_res[1])
    cnn_model.save(os.path.join("results", "FinalModel.h5"))
