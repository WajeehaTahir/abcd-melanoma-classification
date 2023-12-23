# Skin Lesion Classification 

This project focuses on classifying skin lesions into three categories: Common Nevus, Atypical Nevus, and Melanoma. The classification is based on various image properties extracted from dermoscopic images.

## Dataset

We used the PH2 Dataset, which can be accessed [here](https://www.fc.up.pt/addi/ph2%20database.html).

## Workflow

1. **Image List Creation:**
   - Generate a list of lists containing filenames of images, ordered by Common Nevus, Atypical Nevus, and Melanoma.

2. **Class Attribute Definition:**
   - For each class, calculate average values for different properties.
     ![image](https://github.com/WajeehaTahir/abcd-melanoma-classification/assets/88159584/71b59b43-8ce7-438a-a768-4f9b5ce49525)
      
   - Remove the background by cropping out the lesion and calculate the following properties:
      - Symmetry (X and Y)
        
        ![image](https://github.com/WajeehaTahir/abcd-melanoma-classification/assets/88159584/61affe64-17c5-48e4-a51c-462a09a76861) ![image](https://github.com/WajeehaTahir/abcd-melanoma-classification/assets/88159584/556e082a-b4a3-42f2-88e7-8145073d4e34)

      - Circularity
        
        ![image](https://github.com/WajeehaTahir/abcd-melanoma-classification/assets/88159584/19fe5187-5785-47b0-9c66-504dcb9ededa)

      - Diameter
      - Sharpness
        
        ![image](https://github.com/WajeehaTahir/abcd-melanoma-classification/assets/88159584/1edecabc-0816-4e28-88ae-cccc09283c3e)

      - Color Variation
        
        ![image](https://github.com/WajeehaTahir/abcd-melanoma-classification/assets/88159584/7bd63f8b-22df-4997-8ccc-d1b9f083438a)


3. **Scatter Plot Display:**
   - Display the properties of each class in a scatter plot.
     
     ![image](https://github.com/WajeehaTahir/abcd-melanoma-classification/assets/88159584/931bbcb4-addb-45fc-bcec-ad5797d41f81)


4. **Image Classification:**
   - For each image, retrieve properties and calculate the distance with class attributes.
   - Classify the image based on the minimum distance.
     
     ![image](https://github.com/WajeehaTahir/abcd-melanoma-classification/assets/88159584/bf2fecfc-ab65-49ad-b9ef-0e4831b0be69)


5. **Accuracy Calculation:**
   - Maintain counts of correctly and incorrectly identified images.
   - Calculate accuracy
     
     ![image](https://github.com/WajeehaTahir/abcd-melanoma-classification/assets/88159584/6e118fe7-6b54-4906-acdf-3a6c40bc4330)


#
_Documented by ChatGPT_
