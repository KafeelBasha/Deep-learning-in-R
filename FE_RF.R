#Start Time
Sys.time()

#Pre trained model: application_xception()
library(keras)
base_dir="D:\\Deep learning\\Cnn Cont\\Data\\waffle_pancakes"

#Import, resize and extract pixel intensties of images
readimg<-function(files)
{
  imagenet_preprocess_input(image_to_array(image_load(files,target_size =c(224,224)))) #default RGB 
}

#Training set
path="D:\\Deep learning\\Cnn Cont\\Data\\waffle_pancakes\\train"
images=list.files(path,recursive=TRUE,pattern=".png",full.names=TRUE)
length(images)

train=sapply(images,readimg) #pixel intensities
dim(train)

#sapply() arranges values in column wise
train=t(train)
dim(train)

#Column of class labels
head(rownames(train))
labels=sapply(strsplit(rownames(train),"/"),"[",2)
rownames(train)<-NULL
class_train=unlist(labels)

#Test set
path="D:\\Deep learning\\Cnn Cont\\Data\\waffle_pancakes\\test"
images=list.files(path,recursive=TRUE,pattern=".png",full.names=TRUE)
length(images)

test=sapply(images,readimg) #pixel intensities
dim(test)

#sapply() arranges values in column wise
test=t(test)
dim(test)


#Column of class labels
labels_test=sapply(strsplit(rownames(test),"/"),"[",2)
rownames(test)<-NULL
class_test=unlist(labels_test)

#fill the new dimensions in row-major ordering
library(reticulate)
train_x<- array_reshape(train, c(nrow(train),224,224,3))
test_x<- array_reshape(test, c(nrow(test),224,224,3))

train_y=ifelse(class_train=="pancakes",1,0)
test_y=ifelse(class_test=="pancakes",1,0)

rm("train","test")


#Model
base_model=application_xception(include_top = FALSE,pooling="avg",input_shape = c(224,224,3))
train_features=predict(base_model,train_x)
test_features=predict(base_model,test_x)
dim(train_features)
dim(test_features)

colnames(train_features)<-paste0('X',1:ncol(train_features))

colnames(test_features)<-paste0('X',1:ncol(test_features))

library(caret)
library(randomForest)
tune=expand.grid(mtry=c(3))
clf=train(train_features,as.factor(train_y),method="rf",tuneGrid = tune)

#Prediction and accuracy
pred=predict(clf$finalModel,test_features,type="prob")

library(ROCR)
prediction=prediction(pred[,2],test_y)
auc=performance(prediction,"auc")
auc@y.values

#End Time
Sys.time()