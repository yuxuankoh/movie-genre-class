import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
from util_functions import *
import base64

main_bg = "transparent1background.png"
main_bg_ext = "png"
side_bg = "transparent1background.png"
side_bg_ext = "png"

st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
    }}
   .sidebar .sidebar-content {{
        background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()})
    }}
    </style>
    """,
    unsafe_allow_html=True
)

MovieModel = MovieClassificationModel()

st.title("""
         Classification of Movie Genre using Movie Images
         """)
st.write("CZ1105 REP Lab Group 1- Lim Sheng Jie, Ong Hiok Hian, Koh Yu Xuan")
# st.write("Try our Image Clasification Model! Simply upload an image and it will show you the genre!")
st.write("Evaluate how good your movie poster is at conveying the genre of your film! Simply upload your movie poster and leave the rest to us!")

# user_input = st.text_input("Input the most desired genre (Action / Drama/ Thriller / Horror / Comedy): ")
user_input = st.text_input("Input the intended genre of your film (Action / Drama/ Thriller / Horror / Comedy): ")
user_input = user_input.lower()
# possible_genres = ['action', 'comedy', 'drama', 'horror', 'thriller']

# while user_input.lower() not in possible_genres:
#     user_input = st.text_input("Input the intended genre of your film (Action / Drama/ Thriller / Horror / Comedy): ")
if (user_input == 'action'):
    index = 0
elif (user_input == 'comedy'):
    index = 1
elif (user_input == 'drama'):
    index = 2
elif (user_input == 'horror'):
    index = 3
elif (user_input == 'thriller'):
    index = 4
    
if (user_input):
    file = st.file_uploader("Upload an image file", type=["jpg", "jpg"])
    if file:
        preprocess_input = tf.keras.applications.densenet.preprocess_input
        size = (256,256)

        raw_user_image = Image.open(file).convert('RGB') #Open the Image and Convert
        st.image(raw_user_image, use_column_width=True, caption = "Your input poster") #Print the image and show user
        
        image_resized = np.asarray(raw_user_image.resize(size))

        #MODEL CALL
        preds = MovieModel.import_and_predict(np.expand_dims(preprocess_input(image_resized), axis=0))
        preds = preds.tolist()[0]
        for i in preds:
            if i > MovieModel.thresholds[preds.index(i)]:
                st.write(MovieModel.mapBackGenre[preds.index(i)])

        col1, col2, col3 , col4, col5 = st.beta_columns(5)
        with col1:
            pass
        with col2:
            examine_button = st.button("Examine my poster")
        with col4:
            view_alt = st.button("View Alternatives")
        with col5:
            pass
        with col3 :
            pass

        if (examine_button):
            grad_indexes =[preds.index(i) for i in preds if (i > MovieModel.thresholds[preds.index(i)] and preds.index(i) != index)]
            if len(grad_indexes) >0:
                cols = st.beta_columns(len(grad_indexes))
                for j in range(len(cols)):
                    with cols[j]:
                        st.write("Your poster could be mistaken for " + MovieModel.mapBackGenre[grad_indexes[j]])
                        MovieModel.run_gradCAM(img = raw_user_image, pred_index = grad_indexes[j], alpha = 2)
                st.write("Not sure how to change it? Click on View Alternatives for quick suggestions")
            else:
                st.write("Looks Good!")

    # Model that takes in user input
        if (view_alt):
            col1, col2, col3 = st.beta_columns(3)
            if (user_input == 'horror'):
                with col1:
                    st.write("Choice 1")
                    horror1 = Image.open('Horror1.jpg').convert('RGB')
                    #st.image(horror, use_column_width=True)
                    horror1 = np.asarray(horror1)
                    horror1 = horror1.astype(np.float32)[np.newaxis, ...] / 255.
                    MovieModel.style_transfer(raw_user_image, horror1)
                with col2:
                    st.write("Choice 2")
                    horror2 = Image.open('Horror2.jpg').convert('RGB')
                    #st.image(horror, use_column_width=True)
                    horror2 = np.asarray(horror2)
                    horror2 = horror2.astype(np.float32)[np.newaxis, ...] / 255.
                    MovieModel.style_transfer(raw_user_image, horror2)
                with col3:
                    st.write("Choice 3")
                    horror3 = Image.open('Horror3.jpg').convert('RGB')
                    #st.image(horror, use_column_width=True)
                    horror3 = np.asarray(horror3)
                    horror3 = horror3.astype(np.float32)[np.newaxis, ...] / 255.
                    MovieModel.style_transfer(raw_user_image, horror3)

            elif (user_input == 'drama'):
                with col1:
                    st.write("Choice 1")
                    drama1 = Image.open('Drama1.jpg').convert('RGB')
                    #st.image(horror, use_column_width=True)
                    drama1 = np.asarray(drama1)
                    drama1 = drama1.astype(np.float32)[np.newaxis, ...] / 255.
                    MovieModel.style_transfer(raw_user_image, drama1)
                with col2:
                    st.write("Choice 2")
                    drama2 = Image.open('Drama2.jpg').convert('RGB')
                    #st.image(horror, use_column_width=True)
                    drama2 = np.asarray(drama2)
                    drama2 = drama2.astype(np.float32)[np.newaxis, ...] / 255.
                    MovieModel.style_transfer(raw_user_image, drama2)
                with col3:
                    st.write("Choice 3")
                    drama3 = Image.open('Drama2.jpg').convert('RGB')
                    #st.image(horror, use_column_width=True)
                    drama3 = np.asarray(drama3)
                    drama3 = drama3.astype(np.float32)[np.newaxis, ...] / 255.
                    MovieModel.style_transfer(raw_user_image, drama3)

            elif (user_input == 'action'):
                with col1:
                    st.write("Choice 1")
                    action1 = Image.open('Action1.jpg').convert('RGB')
                    #st.image(horror, use_column_width=True)
                    action1 = np.asarray(action1)
                    action1 = action1.astype(np.float32)[np.newaxis, ...] / 255.
                    MovieModel.style_transfer(raw_user_image, action1)
                with col2:
                    st.write("Choice 2")
                    action2 = Image.open('Action2.jpg').convert('RGB')
                    #st.image(horror, use_column_width=True)
                    action2 = np.asarray(action2)
                    action2 = action2.astype(np.float32)[np.newaxis, ...] / 255.
                    MovieModel.style_transfer(raw_user_image, action2)
                with col3:
                    st.write("Choice 3")
                    action3 = Image.open('Action3.jpg').convert('RGB')
                    #st.image(horror, use_column_width=True)
                    action3 = np.asarray(action3)
                    action3 = action3.astype(np.float32)[np.newaxis, ...] / 255.
                    MovieModel.style_transfer(raw_user_image, action3)

            elif (user_input == 'comedy'):
                with col1:
                    st.write("Choice 1")
                    comedy1 = Image.open('Comedy1.jpg').convert('RGB')
                    #st.image(horror, use_column_width=True)
                    comedy1 = np.asarray(comedy1)
                    comedy1 = comedy1.astype(np.float32)[np.newaxis, ...] / 255.
                    MovieModel.style_transfer(raw_user_image, comedy1)
                with col2:
                    st.write("Choice 2")
                    comedy2 = Image.open('Comedy2.jpg').convert('RGB')
                    #st.image(horror, use_column_width=True)
                    comedy2 = np.asarray(comedy2)
                    comedy2 = comedy2.astype(np.float32)[np.newaxis, ...] / 255.
                    MovieModel.style_transfer(raw_user_image, comedy2)
                with col3:
                    st.write("Choice 3")
                    comedy3 = Image.open('Comedy3.jpg').convert('RGB')
                    #st.image(horror, use_column_width=True)
                    comedy3 = np.asarray(comedy3)
                    comedy3 = comedy3.astype(np.float32)[np.newaxis, ...] / 255.
                    MovieModel.style_transfer(raw_user_image, comedy3)

            elif (user_input == 'thriller'):
                with col1:
                    st.write("Choice 1")
                    thriller1 = Image.open('Thriller1.jpg').convert('RGB')
                    #st.image(horror, use_column_width=True)
                    thriller1 = np.asarray(thriller1)
                    thriller1 = thriller1.astype(np.float32)[np.newaxis, ...] / 255.
                    MovieModel.style_transfer(raw_user_image, thriller1)
                with col2:
                    st.write("Choice 2")
                    thriller2 = Image.open('Thriller2.jpg').convert('RGB')
                    #st.image(horror, use_column_width=True)
                    thriller2 = np.asarray(thriller2)
                    thriller2 = thriller2.astype(np.float32)[np.newaxis, ...] / 255.
                    MovieModel.style_transfer(raw_user_image, thriller2)
                with col3:
                    st.write("Choice 3")
                    thriller3 = Image.open('Thriller3.jpg').convert('RGB')
                    #st.image(horror, use_column_width=True)
                    thriller3 = np.asarray(thriller3)
                    thriller3 = thriller3.astype(np.float32)[np.newaxis, ...] / 255.
                    MovieModel.style_transfer(raw_user_image, thriller3)

            
