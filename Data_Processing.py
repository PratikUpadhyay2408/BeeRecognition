import os
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from skimage import io, transform
from sklearn.model_selection import train_test_split

class Data:
    bee_data_path = r'D:\AI\data\bee_data.csv'
    bee_images = r'D:\AI\data\bee_imgs\bee_imgs'

    img_width = 100
    img_height = 100
    img_channels = 3

    def __init__(self, bee_data_path, bee_images, img_width, img_height, img_channels):
        bee_data_path = bee_data_path
        bee_images = bee_images

        img_width = img_width
        img_height = img_height
        img_channels = img_channels

    def load_data(self):
        bee_ref = pd.read_csv(self.bee_data_path, index_col=False)
        bee_ref.dropna(how='any', inplace=True)
        bee_ref = bee_ref[bee_ref['subspecies'] != '-1']
        bee_ref.drop(['caste', 'date', 'time', 'zip code', 'pollen_carrying'], axis=1, inplace=True)
        img_exists = bee_ref['file'].apply(lambda f: os.path.exists(os.path.join(self.bee_images, f)))
        bee_ref = bee_ref[img_exists]
        return bee_ref

    def read_img(self, path):
        img = io.imread(os.path.join(self.bee_images,path))
        img = transform.resize(img, (self.img_width, self.img_height), mode='reflect')
        return img[:, :, :self.img_channels]

    def plot_subspecies(self, bees):
        subspecies = bees['subspecies'].unique()
        f, ax = plt.subplots(nrows=1, ncols=subspecies.size, figsize=(12, 3))
        i = 0
        # Draw the first found bee of given subpecies
        for s in subspecies:
            if s == 'healthy': continue
            file = os.path.join( self.bee_images,  bees[bees['subspecies'] == s].iloc[0]['file'] )
            im = io.imread(file)
            ax[i].imshow(im, resample=True)
            ax[i].set_title(s, fontsize=8)
            i += 1

        plt.suptitle("Subspecies of Bee")
        plt.tight_layout()
        plt.savefig(os.path.join('results', 'bee_species.png'))
        plt.rcParams.update(plt.rcParamsDefault)

    def balance_data_on_category(self, dataset, category, num_rows):
        # balance the dataset based on a category, for a minimum possible bias in the data
        ss_names = dataset[category].unique()
        ss_num = len( ss_names )
        dataset_balanced = None
        for ss in ss_names:
            dataset_cur = dataset[dataset[category] == ss]
            dataset_cur_resampled = sklearn.utils.resample(dataset_cur, n_samples=int(num_rows / ss_num))
            dataset_balanced = pd.concat([dataset_balanced, dataset_cur_resampled])

        return dataset_balanced

    def balanced_data_split(self, bee_ref):
        # Split to train and test before
        bees = bee_ref.sample(frac=1)
        train_bees, test_bees = train_test_split(bees, random_state=24)

        # Train/test rows nums
        n_samples = bees.size / 2
        ratio = 0.25
        test_num = n_samples * ratio
        train_num = n_samples - test_num

        # Resample each subspecies category and add to resulting train dataframe
        train_bees_balanced = self.balance_data_on_category(train_bees, 'subspecies', train_num)
        test_bees_balanced = self.balance_data_on_category(test_bees, 'subspecies', test_num)

        return train_bees_balanced, test_bees_balanced
