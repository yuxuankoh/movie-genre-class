import numpy as np
import tensorflow as tf
# Display
import streamlit as st
import matplotlib.cm as cm
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Flatten
from PIL import Image
import tensorflow_hub as hub
import matplotlib.pyplot as plt

class MovieClassificationModel:
    def __init__(self):
        self.thresholds = {0: 0.5276367, 1: 0.5806225, 2:0.39837518, 3:0.5521859, 4:0.51436293}
        self.size = (256,256)
        self.mapBackGenre =  {0: 'Action', 1: 'Comedy', 2:'Drama', 3:'Horror', 4:'Thriller'}
        self.preprocess_input = tf.keras.applications.densenet.preprocess_input
        self.last_conv_layer_name = 'conv5_block32_2_conv'
        self.hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
        self.model = self.createBaseModel()
        self.gradCAMModel = self.createGradCAMModel(self.model)

    def createBaseModel(self):
        densenet_model = tf.keras.applications.DenseNet169(weights="imagenet", include_top=True, input_tensor=Input(shape=(256, 256, 3)))
        combined_dense1 = Dense(512, activation='relu', kernel_initializer='he_normal')(densenet_model.layers[-2].output)
        combined_drop1 = Dropout(0.5)(combined_dense1)
        combined_dense2 = Dense(256, activation='relu', kernel_initializer='he_normal')(combined_drop1)
        combined_drop2 = Dropout(0.5)(combined_dense2)
        combined_dense3 = Dense(128, activation='relu', kernel_initializer='he_normal')(combined_drop2)
        combined_drop3 = Dropout(0.5)(combined_dense3)
        combined_output = Dense(5, activation='sigmoid', kernel_initializer='glorot_uniform', name = 'combined')(combined_drop3)
        model = Model(inputs=densenet_model.inputs, outputs=combined_output)
        model.load_weights('best_val_f1_1504211951.h5')
        return model

    def createGradCAMModel(self, model):
        gradCAMModel = tf.keras.models.Model(
            [self.model.inputs], [self.model.get_layer(self.last_conv_layer_name).output, self.model.output]
        )
        return gradCAMModel


    def import_and_predict(self, image):
        preds = self.model.predict(image)
        st.write('Prediction Results:')
        return preds


    def style_transfer(self, image, intended):
        image = np.asarray(image)
        image = image.astype(np.float32)[np.newaxis, ...] / 255.

        # Stylize image
        outputs = self.hub_module(tf.constant(image), tf.constant(intended))
        stylized_image = outputs[0]
        fig = plt.figure()
        plt.imshow(stylized_image[0,:,:,:])
        plt.axis('off')
        st.pyplot(fig)


    def make_gradcam_heatmap(self, img_array, pred_index=None):
        # we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = self.gradCAMModel(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        # then sum all the channels to obtain the heatmap class activation
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()

    def get_img_array(self, img_path, size):
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
        array = tf.keras.preprocessing.image.img_to_array(img)
        array = np.expand_dims(array, axis=0)
        return array

    def save_and_display_gradcam(self, img, heatmap, cam_path="cam.jpg", alpha=0.4):
        # Load the original image
        base_img = tf.keras.preprocessing.image.img_to_array(img)
        # Rescale heatmap to a range 0-255
        heatmap = np.uint8(255 * heatmap)

        # Use jet colormap to colorize heatmap
        jet = cm.get_cmap("jet")

        # Use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]
        ###############################################################################################

        st.image(tf.keras.preprocessing.image.array_to_img(jet_heatmap), caption = "the heatmap before resizing looks like:")
        ###############################################################################################

        # Create an image with RGB colorized heatmap
        jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((base_img.shape[1], base_img.shape[0]), Image.BICUBIC)
        jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
        # Superimpose the heatmap on original image
        superimposed_img = jet_heatmap * alpha + base_img
        superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
        # display grad CAM
        st.image(superimposed_img, use_column_width=True, caption = "Highlighted parts show parts that may be problematic")

    def run_gradCAM(self, img, pred_index, alpha = 0.4):
        preprocess_input = tf.keras.applications.densenet.preprocess_input
        # Prepare image
        img_array = np.asarray(img.resize(self.size))
        img_array = preprocess_input(img_array)
        img_array_pred = np.expand_dims(img_array, axis = 0)
        # Remove last layer's sigmoid
        self.model.layers[-1].activation = None
        ###############################################################################################
        st.write("the pixel values before gradCAM is:", img_array)
        ###############################################################################################
        # Generate class activation heatmap
        heatmap = self.make_gradcam_heatmap(img_array_pred, pred_index=pred_index)
        # # Display heatmap
        self.save_and_display_gradcam(img, heatmap, alpha = alpha)


