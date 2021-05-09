import numpy as np
import librosa
import librosa.display
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from time import time
import matplotlib.pyplot as plt

from bss_cnn import BSS_CNN
from musdb_generator import generator

SR = 22050  
WIN = 1024
HOP = 256 
TRGT = 'vocals' 
SMPL = 27  
BS = 64     
TIB = 16    
EPOCHS = 3  
SPE = 6500   

train_gen = generator(SPE, 'train', 'train', TRGT, BS, TIB, SR, WIN, HOP, SMPL)
valid_gen = generator(SPE, 'train', 'valid', TRGT, BS, TIB, SR, WIN, HOP, SMPL)

model = BSS_CNN.define(freq_bins = int(WIN/2+1), length = SMPL)

H = model.fit_generator(generator=train_gen,
                        steps_per_epoch = SPE,
                        epochs = EPOCHS,
                        callbacks = [ ModelCheckpoint('output/model.hdf5', monitor='val_loss', save_best_only=True, save_weights_only=False, mode='auto') ],
                        validation_data = valid_gen,
                        validation_steps = SPE,
                        max_queue_size = 20,
                        workers = 4
                        )
N = np.arange(1, EPOCHS+1)
model.save('output/cnn.hdf5')