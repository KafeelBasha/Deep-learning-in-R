#Start Time
Sys.time()

#Pre trained model:application_inception_v3()
library(keras)
base_dir="D:\\Deep learning\\Cnn Cont\\Data\\waffle_pancakes"

preprocess<-function(x){
  dim(x)<-c(1,dim(x))
  x=imagenet_preprocess_input(x)
}
data_gen=image_data_generator(rotation_range = 40,shear_range=0.2,
                              horizontal_flip = TRUE,vertical_flip = FALSE,
                              zoom_range = 0.2,fill_mode="nearest",
                              preprocessing_function = preprocess)

train_gen=flow_images_from_directory(paste0(base_dir,"\\train"),generator=data_gen,target_size = c(150,150),batch_size = 32)

test_gen=flow_images_from_directory(paste0(base_dir,"\\test"),generator=data_gen,target_size = c(150,150),batch_size = 32)

base_model=application_inception_v3(include_top=FALSE,weights="imagenet",input_shape=c(150,150,3),pooling="avg")
summary(base_model)

x=base_model$output
#Let's add a fully connected layer
x=layer_dense(units=1024,activation="relu")(x)
#Logistic layer with 2 classes
prediction=layer_dense(units=2,activation="softmax")(x)

model=keras_model(inputs=base_model$input,outputs=prediction)
summary(model)

#Freeze weights in a model or layer so that they are no longer trainable
freeze_weights(base_model)
#Compile
model%>%compile(loss="categorical_crossentropy",optimizer="rmsprop",metrics = "accuracy")

fit_generator(model,train_gen,epochs=3,verbose=1,validation_data = test_gen,steps_per_epoch=(947/32),validation_steps = (406/32))

#End Time
Sys.time()