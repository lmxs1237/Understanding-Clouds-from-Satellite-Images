import numpy as np
import pandas as pd
import random

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
#import seaborn as sns

import os
import uuid

from google.cloud import bigquery
from google.oauth2 import service_account


import albumentations as albu
import cv2
import keras
from keras import backend as K
from keras.models import Model
from keras.losses import binary_crossentropy
import numpy as np
import pandas as pd
from keras.models import load_model
from keras import backend as K

credentials = service_account.Credentials.from_service_account_file(
    'credentials json file ')
project_id = 'project_id'
client = bigquery.Client(credentials= credentials,project=project_id)


def rle2mask(rle, input_shape):
    width, height = input_shape[:2]

    mask = np.zeros(width * height).astype(np.uint8)

    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        mask[int(start):int(start + lengths[index])] = 1
        current_position += lengths[index]

    return mask.reshape(height, width).T


def plot_mask_on_img(old_img):

    tdf = pd.read_csv('mask_test.csv', index_col=0)
    tdf = tdf.fillna('')

    plt.figure(figsize=[25, 20])
    types = ''
    for index, row in tdf.iterrows():
        img = cv2.imread(old_img)
        img = cv2.resize(img, (525, 350))
        mask_rle = row['EncodedPixels']
        #     try: # label might not be there!
        #         mask = rle_decode(mask_rle)
        #     except:
        #         mask = np.zeros((1400, 2100))
        plt.subplot(2, 2, index + 1)
        plt.imshow(img)
        if mask_rle != '':
            types = types + ', ' + row['Image_Label'].split('_')[-1]

        plt.imshow(rle2mask(mask_rle, img.shape), alpha=0.5, cmap='gray')
        plt.title(row['Image_Label'].split('_')[-1], fontsize=25)
        plt.axis('off')

    types_all = types[2:]
    after_mask = ''.join(str(uuid.uuid4()).split('-')) + '.png'
    plt.savefig('static/upload/' + after_mask)
    plt.close()

    imgId = str(random.randint(22200,24200))+'.jpg'
    print(imgId)
    for i in types_all.split(', '):
        query = """INSERT INTO `project_id.project_satellite.report`
                   (int64_field_0 , ImageId, Label, exist, Frequent, types )
                   VALUES (2, @d, @a, 1, @b, @c)"""

        query_params = [
            bigquery.ScalarQueryParameter("d", "STRING", imgId),
            bigquery.ScalarQueryParameter("a", "STRING", i),
            bigquery.ScalarQueryParameter("b", "STRING", types_all),
            bigquery.ScalarQueryParameter("c", "INT64", len(types_all.split(', '))),
        ]

        job_config = bigquery.QueryJobConfig()
        job_config.query_parameters = query_params

        client.query(
            query,
            # Location must match that of the dataset(s) referenced in the query.
            location="US",
            job_config=job_config,
        )

    return after_mask


def np_resize(img, input_shape):
    """
    Reshape a numpy array, which is input_shape=(height, width),
    as opposed to input_shape=(width, height) for cv2
    """
    height, width = input_shape
    return cv2.resize(img, (width, height))


def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def build_masks(rles, input_shape, reshape=None):
    depth = len(rles)
    if reshape is None:
        masks = np.zeros((*input_shape, depth))
    else:
        masks = np.zeros((*reshape, depth))

    for i, rle in enumerate(rles):
        if type(rle) is str:
            if reshape is None:
                masks[:, :, i] = rle2mask(rle, input_shape)
            else:
                mask = rle2mask(rle, input_shape)
                reshaped_mask = np_resize(mask, reshape)
                masks[:, :, i] = reshaped_mask

    return masks


def build_rles(masks, reshape=None):
    width, height, depth = masks.shape

    rles = []

    for i in range(depth):
        mask = masks[:, :, i]

        if reshape:
            mask = mask.astype(np.float32)
            mask = np_resize(mask, reshape).astype(np.int64)

        rle = mask2rle(mask)
        rles.append(rle)

    return rles


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score


def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


def segmentation(name,path,model):
    test_df = []
    sub_df = pd.DataFrame(data={'Image_Label': [name+"_Fish", name+"_Flower",
                                                name+"_Gravel", name+"_Sugar"],
                                'EncodedPixels': ['1 1','1 1','1 1','1 1'],'ImageId':[name, name,
                                                                                        name, name]})

    test_imgs=pd.DataFrame(data={'ImageId': [name]})
    test_generator = DataGenerator(
        [0],
        df=test_imgs,
        shuffle=False,
        mode='predict',
        dim=(350, 525),
        reshape=(320, 480),
        n_channels=3,
        base_path=path,
        target_df=sub_df,
        batch_size=1,
        n_classes=4
    )

    batch_pred_masks = model.predict_generator(
        test_generator,
        workers=1,
        verbose=1
    )

    for j, b in enumerate([0]):
        filename = test_imgs['ImageId'].iloc[b]
        image_df = sub_df[sub_df['ImageId'] == filename].copy()

        pred_masks = batch_pred_masks[j, ].round().astype(int)
        pred_rles = build_rles(pred_masks, reshape=(350, 525))

        image_df['EncodedPixels'] = pred_rles
        test_df.append(image_df)

    test_df[0].iloc[:,:2].to_csv('./mask_test.csv')

    return test_df


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, df, target_df=None, mode='fit',
                 base_path='./train_images',
                 batch_size=32, dim=(1400, 2100), n_channels=3, reshape=None,
                 augment=False, n_classes=4, random_state=2019, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.df = df
        self.mode = mode
        self.base_path = base_path
        self.target_df = target_df
        self.list_IDs = list_IDs
        self.reshape = reshape
        self.n_channels = n_channels
        self.augment = augment
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.random_state = random_state

        self.on_epoch_end()
        np.random.seed(self.random_state)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_batch = [self.list_IDs[k] for k in indexes]

        X = self.__generate_X(list_IDs_batch)

        if self.mode == 'fit':
            y = self.__generate_y(list_IDs_batch)

            if self.augment:
                X, y = self.__augment_batch(X, y)

            return X, y

        elif self.mode == 'predict':
            return X

        else:
            raise AttributeError('The mode parameter should be set to "fit" or "predict".')

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.seed(self.random_state)
            np.random.shuffle(self.indexes)

    def __generate_X(self, list_IDs_batch):
        'Generates data containing batch_size samples'
        # Initialization
        if self.reshape is None:
            X = np.empty((self.batch_size, *self.dim, self.n_channels))
        else:
            X = np.empty((self.batch_size, *self.reshape, self.n_channels))

        # Generate data
        for i, ID in enumerate(list_IDs_batch):
            im_name = self.df['ImageId'].iloc[ID]
            img_path = f"{self.base_path}/{im_name}"
            img = self.__load_rgb(img_path)

            if self.reshape is not None:
                img = np_resize(img, self.reshape)

            # Store samples
            X[i,] = img

        return X

    def __generate_y(self, list_IDs_batch):
        if self.reshape is None:
            y = np.empty((self.batch_size, *self.dim, self.n_classes), dtype=int)
        else:
            y = np.empty((self.batch_size, *self.reshape, self.n_classes), dtype=int)

        for i, ID in enumerate(list_IDs_batch):
            im_name = self.df['ImageId'].iloc[ID]
            image_df = self.target_df[self.target_df['ImageId'] == im_name]

            rles = image_df['EncodedPixels'].values

            if self.reshape is not None:
                masks = build_masks(rles, input_shape=self.dim, reshape=self.reshape)
            else:
                masks = build_masks(rles, input_shape=self.dim)

            y[i,] = masks

        return y

    def __load_grayscale(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32) / 255.
        img = np.expand_dims(img, axis=-1)

        return img

    def __load_rgb(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.

        return img

    def __random_transform(self, img, masks):
        composition = albu.Compose([
            albu.HorizontalFlip(),
            albu.VerticalFlip(),
            albu.ShiftScaleRotate(rotate_limit=45, shift_limit=0.15, scale_limit=0.15)
        ])

        composed = composition(image=img, mask=masks)
        aug_img = composed['image']
        aug_masks = composed['mask']

        return aug_img, aug_masks

    def __augment_batch(self, img_batch, masks_batch):
        for i in range(img_batch.shape[0]):
            img_batch[i,], masks_batch[i,] = self.__random_transform(
                img_batch[i,], masks_batch[i,])

        return img_batch, masks_batch


def process(old_img):
    print('start load model')
    model = load_model('./models/unet.h5', custom_objects={'bce_dice_loss': bce_dice_loss, 'dice_coef': dice_coef})
    print('finish load model')

    segmentation(old_img.split('/')[-1], './static/upload',model)
    K.clear_session()


    return plot_mask_on_img(old_img)
