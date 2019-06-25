library(keras)

#Step1: Loading the model in the memory 
model=application_vgg16()
summary(model)

#Step2: Step 2: Load the image, make sure that the image is of size which
#the model can take as an input

im_path="D:\\Deep learning\\Cnn Cont\\Data\\pizza.jpg"
image=image_load(im_path,target_size = c(224,224))
str(image)

#Step3:Preprocess the data
image_array=image_to_array(image)
dim(image_array)
dim(image_array)<- c(1, 224, 224, 3)
dim(image_array)
head(image_array)

image_preproc=imagenet_preprocess_input(image_array)
head(image_preproc)

#Step 4: Use the pre-processed array to obtain predictions from the model
pred=predict(model,image_preproc)
class(pred)
dim(pred)
imagenet_decode_predictions(pred,top=5)

#vgg16 model is trained on imagenet database to identify 1000 common objects/animals/places etc
imagenet_decode_predictions(pred,top=1000)
