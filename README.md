# Predicting-Medical-Tissue-Slices

Invasive ductal carcinoma (IDC) is one of the most common types of breast cancer. It's malicious and able to form metastases which makes it especially dangerous. In the project, we used two pre-trained convolutional neural networks, Resnet18 and VGG16, to detect IDC in the tissue slice images. <br>

## EDA 
**Data Distribution**
![This is an image](Results/distribution.jpeg)
<br>


**Negative Tissue Slices**
![This is an image](Results/negative.jpeg)
<br>

**Positive Tissue Slices**
![This is an image](Results/positive.jpeg)
<br>

## Data Preparation 
In order to avoid the overfitting issue and improve the model performance, we implemented data augmentation methods. 
1. Transformer were used to increase the diversity of the images. 
   
3. Train a general adversarial network (GAN) was implemented to standardize the image. [Go to this GAN repo.](https://github.com/yueli0329/Medical-Image-Generation)
4. Learning rate search method. In the training process, the cyclical learning rate (CLR) search method was used when training the Resnet18 and the VGG16 framework.  The best performance is the accuracy 0.85 on the test set on Resnet18 with CLR search. 

## Training Process


## Results
Image[]
