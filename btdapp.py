#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from keras.models import load_model


# In[2]:


st.title('Brain Tumor Detection')


# In[3]:


# Function to preprocess the input image
def preprocess_image(img):
    img = img.resize((128, 128))
    img_array = np.array(img)
    img_array = img_array.reshape(1, 128, 128, 3)
    return img_array


def names(number):
    if number==0:
        return 'Its a Tumor'
    else:
        return 'No, Its not a tumor'
    


# In[4]:


# Streamlit app
def main():
    st.title("Brain Tumor Detection App")
    st.write("Upload an MRI brain image, and the app will predict whether it has a tumor or not.")

    uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image
        img = Image.open(uploaded_file)

        model=load_model('Brain_Tumor_Model-2.h5')

        x = np.array(img.resize((128,128)))
        x = x.reshape(1,128,128,3)
        res = model.predict_on_batch(x)
        classification = np.where(res == np.amax(res))[1][0]
        print(str(res[0][classification]*100) + '% Confidence This Is A ' + names(classification))

        st.write(str(res[0][classification]*100) + '% Confidence This Is A ' + names(classification))

        # img_array = preprocess_image(img)

        # Make prediction
        # prediction = model.predict(img_array)
        # classification = np.argmax(prediction)

        # Display the prediction result
        # st.write(f"Prediction: {classification}")
        # st.write(f"Confidence: {prediction[0][classification] * 100:.2f}%")


# In[5]:


# Run the app
if __name__ == "__main__":
    main()


# In[ ]:




