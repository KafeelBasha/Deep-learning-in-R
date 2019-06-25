library(keras)
model=application_vgg16()
summary(model)

model1=application_vgg16(include_top=FALSE)
summary(model1)

#Preprocess the data
im_path="D:\\Deep learning\\Cnn Cont\\Data\\pizza.jpg"
image=image_load(im_path,target_size = c(224,224))
str(image)

image_array=image_to_array(image)
dim(image_array)
dim(image_array)<- c(1, 224, 224, 3)
dim(image_array)

image_preproc=imagenet_preprocess_input(image_array)

#Prediction
pred=predict(model1,image_preproc)
class(pred)
dim(pred)

#Visualize the arrays
rotate <- function(x) t(apply(x, 2, rev))

par(mfrow=c(2,2)) #arrange row wise
image(rotate(pred[,,,1]),col=grey.colors(255))
image(rotate(pred[,,,35]),col=grey.colors(255))

#We can think of the intermediate outputs as features, the features extracted by
#the top layers of the model are very abstract and are very difficult to interpret
#we can also extract features from lower layers and visualize

#Subsetting VGG16
model2=keras_model(inputs=model1$input,outputs=get_layer(model1, 'block1_conv1')$output)
model2

#Prediction
pred1=predict(model2,image_preproc)
dim(pred1)                   

#Visualize the arrays
par(mfrow=c(1,2)) #arrange row wise
image(rotate(pred1[,,,1]),col=grey.colors(255))
image(rotate(pred1[,,,2]),col=grey.colors(255))
