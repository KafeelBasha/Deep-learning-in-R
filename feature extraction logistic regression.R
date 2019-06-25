library(keras)
base_dir="D:\\Deep learning\\Cnn Cont\\Data\\waffle_pancakes"

#Import, resize and extract pixel intensties of images
readimg<-function(files)
{
  imagenet_preprocess_input(image_to_array(image_load(files,target_size =c(224,224)))) #default RGB 
}

#Training set
path="D:\\Deep learning\\Cnn Cont\\Data\\waffle_pancakes\\train_sample"
images=list.files(path,recursive=TRUE,pattern=".png",full.names=TRUE)
length(images)

train=sapply(images,readimg) #pixel intensities
dim(train)

#sapply() arranges values in column wise
train=t(train)
dim(train)

#Column of class labels
labels=sapply(strsplit(rownames(train),"/"),"[",2)
rownames(train)<-NULL
class_train=unlist(labels)

#Test set
path="D:\\Deep learning\\Cnn Cont\\Data\\waffle_pancakes\\test_sample"
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
dim(train_features)

#Principal component analysis
colnames(train_features)<-paste0('X',1:ncol(train_features))
prin=prcomp(train_features/255)
names(prin)

dim(prin$rotation)
dim(prin$x)

#compute standard deviation of each principal component
std_dev <- prin$sdev

#compute variance
pr_var <- std_dev^2
pr_var[1:10]

#proportion of variance explained
prop_varex <- pr_var/sum(pr_var)
prop_varex[1:30]

#A scree plot is used to access components or factors which explains
#the most of variability in the data. It represents values in descending order.

#scree plot
plot(prop_varex[1:30], xlab = "Principal Component",
       ylab = "Proportion of Variance Explained",
       type = "b")

#Principal components on Training data
train_pca<-data.frame(prin$x)

#Select first 20 PCs
train_pca<- train_pca[,1:20]

#Decision Tree classifier
library(randomForest)
library(caret)
tune=expand.grid(mtry=c(3))
clf=train(train_pca,as.factor(train_y),method="rf",tuneGrid = tune)

test_features=predict(base_model,test_x)
dim(test_features)

#Principal components on Test data
colnames(test_features)<-paste0('X',1:ncol(test_features))
test_pca=predict(prin,newdata=test_features)

#Select first 20 PCs
test_pca<- test_pca[,1:20]

#Prediction and accuracy
preds=predict(clf$finalModel,newdata=data.frame(test_pca),type="prob")
head(preds)

library(ROCR)
predict=prediction(preds[,2],test_y)
auc=performance(predict,"auc")
auc@y.values
