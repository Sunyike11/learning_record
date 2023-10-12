"""
accuracy: 0.9731
val_accuracy: 0.7333
(过拟合？
"""

import os
import shutil
import tensorflow as tf
from tensorflow.keras import layers

# 分类数据
# source_folder = './data/train1'
# train_folder = './data/train/dogs'
# validation_folder = './data/validation/dogs'
# os.makedirs(train_folder, exist_ok=True)
# os.makedirs(validation_folder, exist_ok=True)
#
# for i in range(1000):
#     source_file = os.path.join(source_folder, f'dog.{i}.jpg')
#     destination_file = os.path.join(train_folder, f'dog.{i}.jpg')
#     shutil.copy(source_file, destination_file)
#
# for i in range(2000, 2500):
#     source_file = os.path.join(source_folder, f'dog.{i}.jpg')
#     destination_file = os.path.join(validation_folder, f'dog.{i}.jpg')
#     shutil.copy(source_file, destination_file)
# print('dogs完成')

# 加载数据集
base_dir = './data/'
train_dir = os.path.join(base_dir, 'train/')
validation_dir = os.path.join(base_dir, 'validation/')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))

num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))

total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val
# print('total_train:', total_train, 'total_val:', total_val)

# 生成训练集和验证集
batchsz = 128
IMG_HEIGHT = 150
IMG_WIDTH = 150
epochs = 15
train_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
validation_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_data_gen = train_image_generator.flow_from_directory(batch_size=batchsz,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')
val_data_gen = validation_image_generator.flow_from_directory(batch_size=batchsz,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')
# 创建模型
model = tf.keras.Sequential([
    layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    # Flatten()的作用是将输入的多维数据（通常是二维或三维数据）转换为一维数据，以便将其馈送到网络中的其他层，如全连接层。
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1)

])

model.compile(optimizer='adam',
              loss=tf.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batchsz,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batchsz
)



