#---DEPENDENCIES--------------------------------------------------------------+

#---general
import matplotlib.pyplot as plt
import numpy as np 
import os
#---tensorflow
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import BatchNormalization
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

#---DATA----------------------------------------------------------------------+

def image_load(path):
    data = []
    for i in os.listdir(path):
        img = image.load_img(path + "\\"+i,target_size=(28,28))
        input_arr = tf.keras.preprocessing.image.img_to_array(img)
        input_arr = np.array([input_arr])
        data.append(input_arr)
    return data

path = "../images"
data = image_load(path)

data = np.array(data)
data = np.reshape(data,(-1,28,28,3))

img_rows = 28
img_cols = 28
channels =3
img_shape = (img_rows,img_cols,channels)

#---ADVERSARIES---------------------------------------------------------------+

#---generator
def build_generator():
    noise_shape = (100,)
    
    model = Sequential()
    model.add(Dense(256, input_shape=noise_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    
    model.add(Dense(np.prod(img_shape), activation='tanh'))
    model.add(Reshape(img_shape))

    model.summary()

    noise = Input(shape=noise_shape)
    img = model(noise)    #Generated image

    return Model(noise, img)

#---discriminator
def build_discriminator():


    model = Sequential()

    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    img = Input(shape=img_shape)
    validity = model(img)

    return Model(img, validity)

#---TRAINING------------------------------------------------------------------+
def train(data, epochs,batch_size=128, save_interval=50):

    X_train = data # loading data
    # Convert to float and Rescale -1 to 1 (Can also do 0 to 1)
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5

    # Add channels dimension. As the input to our gen and discr. has a shape 28x28x1.
    # X_train = np.expand_dims(X_train, axis=3) 

    half_batch = int(batch_size / 2)
    
    for epoch in range(epochs):

        idx = np.random.randint(0, X_train.shape[0], half_batch)
        imgs = X_train[idx]

 
        noise = np.random.normal(0, 1, (half_batch, 100))

        # Generate a half batch of fake images
        gen_imgs = generator.predict(noise)
        # print(gen_imgs.shape)

        # Train the discriminator on real and fake images, separately
        # Research showed that separate training is more effective. 
        d_loss_real = discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
        # take average loss from real and fake images. 
    
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake) 
        
        noise = np.random.normal(0, 1, (batch_size, 100)) 

        # The generator wants the discriminator to label the generated samples
        # as valid (ones)
        # This is where the genrator is trying to trick discriminator into believing
        # the generated image is true (hence value of 1 for y)
        valid_y = np.array([1] * batch_size) #Creates an array of all ones of size=batch size

        # Generator is part of combined where it got directly linked with the discriminator
        # Train the generator with noise as x and 1 as y. 
        # Again, 1 as the output as it is adversarial and if generator did a great
        # job of folling the discriminator then the output would be 1 (true)
        g_loss = combined.train_on_batch(noise, valid_y)

        
        print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

        # If at save interval => save generated image samples
        if epoch % save_interval == 0:
            save_imgs(epoch)
            
def save_imgs(epoch):
    # r, c = 2,2
    noise = np.random.normal(0, 1, (1, 100))
    gen_imgs = generator.predict(noise)
    gen_imgs = np.reshape(gen_imgs,(28,28,3))
    b = image.array_to_img(gen_imgs)
    
    b.save("images/mnist_%d.png" % epoch)
    plt.close()

optimizer = Adam(0.0002, 0.5)  #Learning rate and momentum.

# Build and compile the discriminator first. 
# Generator will be trained as part of the combined model, later. 
# pick the loss function and the type of metric to keep track.                 
# Binary cross entropy as we are doing prediction and it is a better
# loss function compared to MSE or other. 
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy'])

# build and compile our Discriminator, pick the loss function

# Since we are only generating (faking) images, let us not track any metrics.
generator = build_generator()
generator.compile(loss='binary_crossentropy', optimizer=optimizer)

z = Input(shape=(100,))   # Our random input to the generator
img = generator(z)
discriminator.trainable = False
valid = discriminator(img)
combined = Model(z, valid)
combined.compile(loss='binary_crossentropy', optimizer=optimizer)
train(data,epochs=200001, batch_size=32, save_interval=10000)

generator.save('generator.h5')

