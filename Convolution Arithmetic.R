#Realtionship between output size and padding, strides and kernel size

#Determing output size
size_out=function(input,padding,kernel,strides){
  ((input+2*padding-kernel)/strides)+1
}

size_out(input=5,padding=2,kernel=4,strides=1)

size_k=function(input,padding,strides,output)
{
  input+2*padding+(1-output)*strides
}

size_k(input=5,padding=2,strides=1,output=6)

library(keras)
#First Convolution
size_k(input=32,padding=0,strides=1,output=28)

#First Pooling
size_k(input=28,padding=0,strides=2,output=14)

#Second Convolution
size_k(input=14,padding=0,strides=1,output=10)

#Second Pooling
size_k(input=10,padding=0,strides=2,output=5)

  #Convolutional neural network architechture
model=keras_model_sequential()

model%>%layer_conv_2d(filters=6,kernel_size=c(5,5),padding="valid",strides=1,activation="relu",input_shape=c(32,32,1))%>%
  layer_max_pooling_2d(pool_size=c(2,2),strides=2)%>%
  layer_conv_2d(filters=16,kernel_size=c(5,5),padding="valid",strides=1,activation="relu")%>%
  layer_max_pooling_2d(pool_size=c(2,2),strides=2)%>%
  layer_flatten()%>%
  layer_dense(120,activation="relu")%>%layer_dense(84,activation="relu")%>%
  layer_dense(10,activation="softmax")

summary(model)
