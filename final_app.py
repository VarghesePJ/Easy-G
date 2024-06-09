from skimage.io import imread
from skimage import color
import matplotlib.pyplot as plt
import streamlit as st
from skimage.filters import threshold_otsu,gaussian
from skimage.transform import resize
from numpy import asarray
from skimage.metrics import structural_similarity
from skimage import measure
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import joblib
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import os
from natsort import natsorted

# Function to display ECG details
def show_ecg_details():
    st.header("ECG Details")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a file")
    
    if uploaded_file is not None:
        # Read uploaded image
        image = imread(uploaded_file)
        image_gray = color.rgb2gray(image)
        image_gray = resize(image_gray, (1572, 2213))
        """#### **UPLOADED ECG IMAGE**"""
  
        #checkign if we parse the user image and similar to our format
        image1=imread("C:\\Users\\vargh\\Desktop\\Easy_G\\test\\PMI(87).jpg")
        image1 = color.rgb2gray(image1)
        image1=resize(image1,(1572,2213))

        image2=imread("C:\\Users\\vargh\\Desktop\\Easy_G\\test\\HB(87).jpg")
        image2 = color.rgb2gray(image2)
        image2=resize(image2,(1572,2213))

        image3=imread("C:\\Users\\vargh\\Desktop\\Easy_G\\test\\N(278).jpg")
        image3 = color.rgb2gray(image3)
        image3=resize(image2,(1572,2213))

        image4=imread("C:\\Users\\vargh\\Desktop\\Easy_G\\test\\MI(114).jpg")
        image4 = color.rgb2gray(image4)
        image4=resize(image2,(1572,2213))

        similarity_score = max(
          structural_similarity(image_gray, image1, data_range=image_gray.max() - image_gray.min()),
          structural_similarity(image_gray, image2, data_range=image_gray.max() - image_gray.min()),
          structural_similarity(image_gray, image3, data_range=image_gray.max() - image_gray.min()),
          structural_similarity(image_gray, image4, data_range=image_gray.max() - image_gray.min()))

        if similarity_score > 0.70:
          st.image(image)
        #"""#### **GRAY SCALE IMAGE**"""
        my_expander = st.expander(label='Gray SCALE IMAGE')
        #with my_expander: 
        #  st.image(image_gray)
          
        #"""#### **DIVIDING LEADS**"""
        #dividing the ECG leads from 1-13 from the above image
        Lead_1 = image[300:600, 150:643]
        Lead_2 = image[300:600, 646:1135]
        Lead_3 = image[300:600, 1140:1625]
        Lead_4 = image[300:600, 1630:2125]
        Lead_5 = image[600:900, 150:643]
        Lead_6 = image[600:900, 646:1135]
        Lead_7 = image[600:900, 1140:1625]
        Lead_8 = image[600:900, 1630:2125]
        Lead_9 = image[900:1200, 150:643]
        Lead_10 = image[900:1200, 646:1135]
        Lead_11 = image[900:1200, 1140:1625]
        Lead_12 = image[900:1200, 1630:2125]
        Lead_13 = image[1250:1480, 150:2125]
        Leads=[Lead_1,Lead_2,Lead_3,Lead_4,Lead_5,Lead_6,Lead_7,Lead_8,Lead_9,Lead_10,Lead_11,Lead_12,Lead_13]
        #plotting lead 1-12
        fig , ax = plt.subplots(4,3)
        fig.set_size_inches(10, 10)
        x_counter=0
        y_counter=0

        for x,y in enumerate(Leads[:len(Leads)-1]):
          if (x+1)%3==0:
            ax[x_counter][y_counter].imshow(y)
            ax[x_counter][y_counter].axis('off')
            ax[x_counter][y_counter].set_title("Leads {}".format(x+1))
            x_counter+=1
            y_counter=0
          else:
            ax[x_counter][y_counter].imshow(y)
            ax[x_counter][y_counter].axis('off')
            ax[x_counter][y_counter].set_title("Leads {}".format(x+1))
            y_counter+=1
    
        fig.savefig('Leads_1-12_figure.png')
        fig1 , ax1 = plt.subplots()
        fig1.set_size_inches(10, 10)
        ax1.imshow(Lead_13)
        ax1.set_title("Leads 13")
        ax1.axis('off')
        fig1.savefig('Long_Lead_13_figure.png')
        my_expander1 = st.expander(label='DIVIDING LEAD')
        #with my_expander1:
        #  st.image('Leads_1-12_figure.png')
        #  st.image('Long_Lead_13_figure.png')

        #"""#### **PREPROCESSED LEADS**"""
        fig2 , ax2 = plt.subplots(4,3)
        fig2.set_size_inches(10, 10)
        #setting counter for plotting based on value
        x_counter=0
        y_counter=0

        for x,y in enumerate(Leads[:len(Leads)-1]):
          #converting to gray scale
          grayscale = color.rgb2gray(y)
          #smoothing image
          blurred_image = gaussian(grayscale, sigma=0.9)
          #thresholding to distinguish foreground and background
          #using otsu thresholding for getting threshold value
          global_thresh = threshold_otsu(blurred_image)

          #creating binary image based on threshold
          binary_global = blurred_image < global_thresh
          #resize image
          binary_global = resize(binary_global, (300, 450))
          if (x+1)%3==0:
            ax2[x_counter][y_counter].imshow(binary_global,cmap="gray")
            ax2[x_counter][y_counter].axis('off')
            ax2[x_counter][y_counter].set_title("pre-processed Leads {} image".format(x+1))
            x_counter+=1
            y_counter=0
          else:
            ax2[x_counter][y_counter].imshow(binary_global,cmap="gray")
            ax2[x_counter][y_counter].axis('off')
            ax2[x_counter][y_counter].set_title("pre-processed Leads {} image".format(x+1))
            y_counter+=1
        fig2.savefig('Preprossed_Leads_1-12_figure.png')
    
        #plotting lead 13
        fig3 , ax3 = plt.subplots()
        fig3.set_size_inches(10, 10)
        #converting to gray scale
        grayscale = color.rgb2gray(Lead_13)
        #smoothing image
        blurred_image = gaussian(grayscale, sigma=0.7)
        #thresholding to distinguish foreground and background
        #using otsu thresholding for getting threshold value
        global_thresh = threshold_otsu(blurred_image)
        print(global_thresh)
        #creating binary image based on threshold
        binary_global = blurred_image < global_thresh
        ax3.imshow(binary_global,cmap='gray')
        ax3.set_title("Leads 13")
        ax3.axis('off')
        fig3.savefig('Preprossed_Leads_13_figure.png')

        my_expander2 = st.expander(label='PREPROCESSED LEAD')
        #with my_expander2:
        #  st.image('Preprossed_Leads_1-12_figure.png')
        #  st.image('Preprossed_Leads_13_figure.png')
    
        #"""#### **EXTRACTING SIGNALS(1-13)**"""
        fig4 , ax4 = plt.subplots(4,3)
        fig4.set_size_inches(10, 10)
        x_counter=0
        y_counter=0
        for x,y in enumerate(Leads[:len(Leads)-1]):
          #converting to gray scale
          grayscale = color.rgb2gray(y)
          #smoothing image
          blurred_image = gaussian(grayscale, sigma=0.9)
          #thresholding to distinguish foreground and background
          #using otsu thresholding for getting threshold value
          global_thresh = threshold_otsu(blurred_image)

          #creating binary image based on threshold
          binary_global = blurred_image < global_thresh
          #resize image
          binary_global = resize(binary_global, (300, 450))
          #finding contours
          contours = measure.find_contours(binary_global,0.8)
          contours_shape = sorted([x.shape for x in contours])[::-1][0:1]
          for contour in contours:
            if contour.shape in contours_shape:
              test = resize(contour, (255, 2))
          if (x+1)%3==0:
            ax4[x_counter][y_counter].plot(test[:, 1], test[:, 0],linewidth=1,color='black')
            ax4[x_counter][y_counter].axis('image')
            ax4[x_counter][y_counter].set_title("Contour {} image".format(x+1))
            x_counter+=1
            y_counter=0
          else:
            ax4[x_counter][y_counter].plot(test[:, 1], test[:, 0],linewidth=1,color='black')
            ax4[x_counter][y_counter].axis('image')
            ax4[x_counter][y_counter].set_title("Contour {} image".format(x+1))
            y_counter+=1
    
          #scaling the data and testing
          lead_no=x
          scaler = MinMaxScaler()
          fit_transform_data = scaler.fit_transform(test)
          Normalized_Scaled=pd.DataFrame(fit_transform_data[:,0], columns = ['X'])
          Normalized_Scaled=Normalized_Scaled.T
          #scaled_data to CSV
          if (os.path.isfile('scaled_data_1D_{lead_no}.csv'.format(lead_no=lead_no+1))):
            Normalized_Scaled.to_csv('Scaled_1DLead_{lead_no}.csv'.format(lead_no=lead_no+1), mode='a',index=False)
          else:
            Normalized_Scaled.to_csv('Scaled_1DLead_{lead_no}.csv'.format(lead_no=lead_no+1),index=False)
    
        fig4.savefig('Contour_Leads_1-12_figure.png')
        my_expander3 = st.expander(label='CONOTUR LEADS')
        #with my_expander3:
        #  st.image('Contour_Leads_1-12_figure.png')

        #"""#### **CONVERTING TO 1D SIGNAL**"""   
        # Let's try combining all 12 leads
        test_final = pd.read_csv("C:\\Users\\vargh\\Desktop\\Easy_G\\Scaled_1DLead_1.csv")
        location = "C:\\Users\\vargh\\Desktop\\Easy_G\\"
        for files in natsorted(os.listdir(location)):
            if files.endswith(".csv"):
                if files != 'Scaled_1DLead_1.csv':
                    df = pd.read_csv("C:\\Users\\vargh\\Desktop\\Easy_G\\{}".format(files))
                    test_final = pd.concat([test_final, df], axis=1, ignore_index=True)
        #st.write(test_final)
   
        """#### **PREDICTION**"""
        loaded_model = joblib.load("C:\\Users\\vargh\\Desktop\\Easy_G\\file\\model_test.pkl")
        result = loaded_model.predict(test_final)
        if result[0] == 0:
          st.write("You ECG corresponds to Abnormal Heartbeat")

        if result[0] == 1:
          st.write("You ECG corresponds to Myocardial Infarction")

        if result[0] == 2:
          st.write("Your ECG is History of Myocardial Infarction")

        if result[0] == 3:
          st.write("You ECG corresponds to Normal")


# Function to display sugar and pressure level details
def show_sugar_pressure_details():
    st.header("Sugar and Pressure Level Details")
    # Add code to display sugar and pressure level details here
    
def analyze_sugar_level(sugar_level):
    if sugar_level <= 70:
        return "Low Sugar", "Increase sugar intake by consuming fruits or glucose tablets."
    elif 70 < sugar_level <= 160:
        return "Normal Sugar", "Maintain a balanced diet and regular exercise."
    else:
        return "High Sugar", "Reduce sugar intake by avoiding sugary drinks and refined carbohydrates."

def analyze_pressure_level(systolic_pressure, diastolic_pressure):
    if systolic_pressure < 90 or diastolic_pressure < 60:
        return "Low Blood Pressure", "Increase salt intake and stay hydrated."
    elif 90 <= systolic_pressure <= 120 and 60 <= diastolic_pressure <= 80:
        return "Normal Blood Pressure", "Maintain a healthy lifestyle with balanced diet and regular exercise."
    elif 120 < systolic_pressure <= 140 or 80 < diastolic_pressure <= 90:
        return "Prehypertension", "Monitor blood pressure regularly and consider lifestyle changes."
    else:
        return "High Blood Pressure", "Reduce salt intake, exercise regularly, and consult a doctor if necessary."

def show_sugar_pressure_details():
    st.header("Sugar and Pressure Level Details")

    """#### **Sugar Level**"""
    
    # Input for sugar level
    sugar_level = st.number_input("Enter Sugar Level (mg/dL):", min_value=0.0)

    # Analyze sugar level
    sugar_category, sugar_tip = analyze_sugar_level(sugar_level)

    # Display sugar level result and tip
    st.write("Sugar Level:", sugar_level, "mg/dL")
    st.write("Sugar Result:", sugar_category)
    st.write("Sugar Tip:", sugar_tip)

    """#### **Pressure Level**"""

    # Input for blood pressure
    systolic_pressure = st.number_input("Enter Systolic Pressure (mmHg):", min_value=0)
    diastolic_pressure = st.number_input("Enter Diastolic Pressure (mmHg):", min_value=0)
    
    # Analyze blood pressure
    pressure_category, pressure_tip = analyze_pressure_level(systolic_pressure, diastolic_pressure)
    
    # Display blood pressure result and tip
    st.write("Systolic Pressure:", systolic_pressure, "mmHg")
    st.write("Diastolic Pressure:", diastolic_pressure, "mmHg")
    st.write("Pressure Result:", pressure_category)
    st.write("Pressure Tip:", pressure_tip)

def main():
    st.title("Easy-G : Health Monitoring System")
    option = st.sidebar.selectbox("Select Option", ["ECG Details", "Sugar and Pressure Details"])

    if option == "ECG Details":
        show_ecg_details()
    elif option == "Sugar and Pressure Details":
        show_sugar_pressure_details()

if __name__ == "__main__":
    main()
