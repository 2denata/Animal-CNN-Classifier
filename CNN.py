# -*- coding: utf-8 -*-
"""215314107_CNN_Jean Paul Denata.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ywm3gcp7AIBMvfZJbqf-hg1ur_tIcnS_

## Nama : Jean Paul Denata
## NIM : 215314107

### Dataset : https://drive.google.com/file/d/1a_oH1olkINKoJjhngNe8aMeEi3-xSqUp/view?usp=sharing
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Inisiasi
MesinKlasifikasi = Sequential()

"""### Membangun Model"""

# Conv2D dan Max Pooling

MesinKlasifikasi.add(Conv2D(32, (3, 3), activation = 'relu'))
MesinKlasifikasi.add(MaxPooling2D(pool_size = (2, 2)))

# Flatten

MesinKlasifikasi.add(Flatten())

# Full Connection

MesinKlasifikasi.add(Dense(units = 120, activation = 'relu'))
MesinKlasifikasi.add(Dense(units = 3, activation = 'softmax'))

# Menjalankan CNN

MesinKlasifikasi.compile(optimizer = 'adam',
                         loss='categorical_crossentropy',
                         metrics = ['accuracy'])

"""### Reshape gambar agar ukuran sama semua"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True
                                  )

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('datasets/training_set',
                                                target_size = (128, 128),
                                                batch_size = 32,
                                                class_mode = 'categorical'
                                                )

test_set = test_datagen.flow_from_directory('datasets/test_set',
                                                target_size = (128, 128),
                                                batch_size = 32,
                                                class_mode = 'categorical'
                                                )

MesinKlasifikasi.fit_generator(training_set,
                                steps_per_epoch = 8300/32,
                                epochs = 25, #harusnya 50 karna berat
                                validation_data = test_set,
                                validation_steps = 2100/32
                              )

"""## Mengecek Kelas"""

training_set.class_indices

"""## Prediksi Kelas

##### 1000 gambar berisi 3 kelas hewan (cat, dog, tiger)
"""

import numpy as np
from keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt

# Variabel untuk menghitung jumlah masing-masing hewan
count_dog = 0
count_cat = 0
count_tiger = 0

# Cek tiap file testing dari foto ke-1 sampai 1000
for i in range(1, 1001):

    # Ambil file dari direktori lalu prediksi memakai model
    test_image = load_img('datasets/test_set/3_class_test/animal (' + str(i) + ').jpg', target_size = (128, 128))
    test_image = img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = MesinKlasifikasi.predict(test_image)

    # Cek, jika prediksi 0 maka klasifikasinya kucing
    if result[0][0] == 0:
        prediction = 'cat'
        print(prediction,i)
        count_cat = count_cat + 1

    # jika prediksi 1 maka klasifikasinya anjing
    elif result[0][0] == 1:
        prediction = 'dog'
        print(result[0][0])
        print(prediction,i)
        count_dog = count_dog + 1

    # jika prediksi 2 maka klasifikasinya tiger
    else:
        prediction = 'tiger'
        print(prediction,i)
        count_tiger = count_tiger + 1

"""#### Hitung jumlah hewan yang telah diklasifikasi"""

print('Jumlah anjing : ', count_dog)
print('Jumlah kucing : ', count_cat)
print('Jumlah tiger : ', count_tiger)

"""## Uji Data Tunggal

### Kucing
"""

print('Gambar kucing: ')

# Ambil direktori dari file gambar dog
direktori_img_kucing = 'cat.4210.jpg'
kucing = load_img('datasets/test_set/cats/' + direktori_img_kucing, target_size = (128, 128))

# Tampilkan gambar kucing
plt.imshow(kucing)
plt.show()

# Ubah gambar ke array, lalu predict dari model
kucing_array = img_to_array(kucing)
kucing_array = np.expand_dims(kucing_array, axis = 0)
result = MesinKlasifikasi.predict(kucing_array)

# Mendeskripsikan hasil klasifikasi
if result[0][0] == 0:
    print('Gambar dari file ' + direktori_img_kucing + ' adalah kucing')
elif result[0][0] == 1:
    print('Gambar dari file ' + direktori_img_kucing + ' adalah anjing')
else:
    print('Gambar dari file ' + direktori_img_kucing + ' adalah tiger')

"""### Anjing"""

print('Gambar anjing: ')

# Ambil direktori dari file gambar dog
direktori_img_anjing = 'dog.4314.jpg'
anjing = load_img('datasets/test_set/dogs/' + direktori_img_anjing, target_size = (128, 128))

# Tampilkan gambar anjing
plt.imshow(anjing)
plt.show()

# Ubah gambar ke array, lalu predict dari model
anjing_array = img_to_array(anjing)
anjing_array = np.expand_dims(anjing_array, axis = 0)
result = MesinKlasifikasi.predict(anjing_array)

# Menampilkan deskripsi
if result[0][0] == 0:
    print('Gambar dari file ' + direktori_img_anjing + 'adalah kucing')
elif result[0][0] == 1:
    print('Gambar dari file ' + direktori_img_anjing + ' adalah anjing')
else:
    print('Gambar dari file ' + direktori_img_anjing + ' adalah tiger')

"""### Tiger"""

print('Gambar tiger: ')

# Ambil direktori dari file gambar tiger
direktori_img_tiger = 'tiger (4005).jpg'
tiger = load_img('datasets/test_set/tigers/' + direktori_img_tiger, target_size = (128, 128))

# Tampilkan gambar
plt.imshow(tiger)
plt.show()

# Ubah gambar ke array, lalu predict dari model
tiger_array = img_to_array(tiger)
tiger_array = np.expand_dims(tiger, axis = 0)
result = MesinKlasifikasi.predict(tiger_array)

# Menampilkan deskripsi
if result[0][0] == 0:
    print('Gambar dari file ' + direktori_img_tiger + ' adalah kucing')
elif result[0][0] == 1:
    print('Gambar dari file ' + direktori_img_tiger + ' adalah anjing')
else:
    print('Gambar dari file ' + direktori_img_tiger + ' adalah tiger')

