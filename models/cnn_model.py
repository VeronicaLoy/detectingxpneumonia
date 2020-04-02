from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Dropout,Flatten,BatchNormalization,MaxPooling2D,Activation,GlobalAveragePooling2D
from tensorflow.keras.layers import Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as k
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import logging
from PIL import Image
import numpy as np

class PneumoniaPrediction:
    def __init__(self):
        '''
        Initialize Class.
        '''
        self.img_generator_train = None
        self.img_generator_test = None
        self.train_generator = None
        self.test_generator = None
        self.valid_generator = None
        self.model = None
        
        return
    
    def set_directory(self, folderpath, 
                      batch_size=16, 
                      rotation_range=20,
                      width_shift_range=0.2,
                      height_shift_range=0.2,
                      horizontal_flip=True,
                      rescale=1/255,
                      validation_split=0.2,
                      target_size=(224, 224)):
        
        '''
        Set a flow from directory for training. Not
        needed for prediction. 
        
        Set folderpath as the path to your training images.
        Image folder need to have train/test/val sub folders
        
        Flow from directory seed is set to 42.
        
        '''
        
        logging.info('Creating image generator...')
        self.img_generator_train = ImageDataGenerator(rotation_range=rotation_range, 
                                                      width_shift_range=width_shift_range, 
                                                      height_shift_range=height_shift_range,
                                                      horizontal_flip=horizontal_flip,
                                                      rescale=rescale,
                                                      validation_split=validation_split)
        
        self.img_generator_test = ImageDataGenerator(rescale=1/255) # seperate generator with no augmentation
        
        self.train_generator = self.img_generator_train.flow_from_directory(
            directory=folderpath+'/train',
            target_size=target_size,
            color_mode="rgb",
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True,
            seed=42,
        )
        
        self.valid_generator = self.img_generator_test.flow_from_directory(
            directory=folderpath+'/val',
            target_size=target_size,
            color_mode="rgb",
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True,
            seed=42,
        )

        self.test_generator = self.img_generator_test.flow_from_directory(
            directory=folderpath+'/test',
            target_size=target_size,
            color_mode="rgb",
            batch_size=1,
            class_mode='categorical',
            shuffle=True,
            seed=42,
        )
        
        logging.info('Image generator created')
        return
    
    def load_img(self, img_path=None):
        '''
        Load image. Please provide path to image
        '''
        if img_path is None:
            logging.warning('No image path defined')
            raise ValueError('No image path defined')
        else:
            logging.info('Loading image...')
            img = Image.open(img_path)
            logging.info('Image loaded')
            return img
    
    def resize(self, img=None):
        '''
        Load and resize image.
        Returns the np array of resized image.
        '''
        if img is None:
            logging.warning('No image defined for resizing')
            raise ValueError('No image fed')
        logging.info('Resizing image...')
        if type(img) != np.ndarray:
            img_new = np.array(img.resize((224, 224))).reshape(-1, 224, 224, 1) / 255
        else:
            # if len(img.shape) > 2:
                # img = img[:, :, 0]
            # print(img.shape)
            img_new = img.reshape((-1, 224, 224, 1)) / 255
        logging.info('Image resized')
        return img_new
    
    def train(self, save_model_name, kwargs):
        '''
        Used to train model.
        Input number of epochs to end training.
        kwargs used for keras' fit function.
        '''
        if self.model is None:
            raise ValueError('No model created. Please create a model first')
        
        es = EarlyStopping(monitor='val_loss', verbose=1, patience=5)
        print('Created early stopping...')
        cp = ModelCheckpoint(save_model_name, verbose=1, save_best_only=True)
        print('Created model checkpoint...')
        self.model.fit(self.train_generator, validation_data=self.test_generator, 
                       callbacks=[es, cp], **kwargs)
        return
    
    def predict(self, img):
        '''
        Predict what the image is. Returns
        the list of probabilities.
        Feed the image path as the first arg.
        '''
        if self.model is None:
            logging.warning('No model initialized for prediction. Returning -1...')
            raise ValueError('Model not loaded')
        img = self.resize(img)
        pred = self.model.predict(img)
        return pred
    
    def load_model(self, model_path):
        '''
        Load keras model.
        Give it the model path.
        '''
        k.clear_session()
        self.model = load_model(model_path)
        return
    
    def load_weights(self, weights_path):
        '''
        Load model weights.
        Require initial model to be created.
        '''
        if self.model is None:
            print('No model created to load weights')
            raise ValueError('No model to load weights to')
        self.model.load_weights(weights_path)
        return
    
    def create_model(self):
        model = Sequential()
        model.add(Conv2D(64,(3,3), activation='relu', input_shape=(224, 224, 3)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(BatchNormalization())
        # model.add(Dropout(0.15))
        model.add(Conv2D(128,(3, 3),activation='relu'))
        model.add(Conv2D(128,(3, 3),activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(BatchNormalization())
        # model.add(Dropout(0.15))
        model.add(Conv2D(256,(3,3),activation='relu'))
        model.add(Conv2D(256,(3,3),activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(BatchNormalization())
        # model.add(Dropout(0.15))
        model.add(Conv2D(512,(3,3),activation='relu'))
        model.add(Conv2D(512,(3,3),activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(BatchNormalization())
        model.add(Dropout(0.15))
        model.add(GlobalAveragePooling2D())
        model.add(Dense(1024, activation='relu'))
        # model.add(Dense(1,activation='sigmoid'))
        model.add(Dense(2,activation='softmax'))
        model.compile(optimizer='adam', 
                      loss = 'categorical_crossentropy', 
                      metrics=['accuracy'])
        
        self.model = model
        return
    