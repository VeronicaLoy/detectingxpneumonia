import time

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow import keras
from keras.preprocessing import image
from keras.preprocessing.image import load_img
from lime import lime_image
from skimage import img_as_float
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt


def intro():
    st.title('Detecting Pneumonia from X-Rays')
    st.write(
    f'''Pneumonia is the leading cause of death for children under 5 — more than 800,000 children died from pneumonia in 2017.
    While common, accurately diagnosing pneumonia through X-Rays is difficult, requires highly trained specialist, and can be time consuming.
    Here, we developed a model which can detect pneumonia from chest radiograph, and a simple UI which allows you to peek into what the model
    looked at in coming up with the diagnosis.
    ''')


def about_model():
    st.header('**_About the model_**')
    st.write(
    f'''xxx'''
    )
    st.header('**_Dataset_**')


def meet_the_team():
    st.header('**_Meet the Team_**')
    st.write(
    '''Hello. This app is developed by a team of apprentices from AISG. We have superior googling and Ctrl C skills. Special thanks to Keras, LIME and Streamlit.
    
    Hire us maybe? 
    '''
    )



def toggle_bar():
    # LIME
    ## Choosing parameters for LIME through drop-down box
    parameters = pd.DataFrame({
    'label': [0, 1],
    'positive_only': [True, False],
    'negative_only': [True, False],
    'hide_rest': [True,False]
    })

    label_option = st.sidebar.selectbox(
    'label', parameters['label'])

    positive_only_option = st.sidebar.selectbox(
    'positive_only', parameters['positive_only'])

    negative_only_option = st.sidebar.selectbox(
    'negative_only', parameters['negative_only'])

    hide_rest_option = st.sidebar.selectbox(
    'hide_rest', parameters['hide_rest'])




def main():

    model = keras.models.load_model("models/best_model_2.h5")

    # Intro
    intro()

    st.sidebar.image('logo.png')
    #st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Menu",
        ["Try it out!", "About the model", "Meet the Team"])
    if app_mode == "Try it out!":
        st.header('**_Try it out!_**')
        upload_file()
    elif app_mode == "About the model":
        about_model()
    elif app_mode == "Meet the Team":
        meet_the_team()


 


def upload_file():
    # Model Prediction
    uploaded_file = st.file_uploader("Choose an image...") #, type="jpg"
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        preprocessing_grayscale(uploaded_file)



    




    toggle_bar()


    
    if uploaded_file is not None:
        # preprocess and explain image here <code from jupyter>

        def predict_fn(x):
            x = np.expand_dims(img_array, axis=0)
            return model.predict(x)

        img_gray = load_img(uploaded_file, color_mode='grayscale', target_size=(224, 224))
        img_array = image.img_to_array(img_gray, dtype='float32')
        explainer = lime_image.LimeImageExplainer() 
        exp = explainer.explain_instance(np.array(img_gray), predict_fn, top_labels=1, hide_color=0, batch_size=1, num_samples=5) 
        exp.top_labels[0]
        temp, mask = exp.get_image_and_mask(0, positive_only=False, num_features=5, hide_rest=False) # change 0 to dynamic

        fig, ax = plt.subplots()
        im = ax.imshow(mark_boundaries(temp, mask))
        st.pyplot()

        st.write('Done')

        #plt.show()




def preprocessing_grayscale(uploaded_file):
    # preprocess and predict image here <code from jupyter>
    img_raw = load_img(uploaded_file, color_mode='grayscale', target_size=(224, 224))
    img_array = image.img_to_array(img_raw, dtype='float32')
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    
    st.write('Prediction:', pred[0][0])



def preprocessing_rgb():





    return None


if __name__ == "__main__":
    main()

