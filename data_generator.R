base_dir="D:\\Deep learning\\Cnn Cont\\Data\\waffle_pancakes"
files=list.files(paste0(base_dir,"\\train\\waffles"),full.names=TRUE)
head(files)
length(files)

#Data Augmentation
library(keras)
gen=image_data_generator(vertical_flip = FALSE)
train_gen=flow_images_from_directory(paste0(base_dir,"\\train"),generator = gen,batch_size=32,
          save_to_dir=paste0(base_dir,"\\augumented_images"),shuffle = FALSE,
          target_size = c(224,224))
class(train_gen)

#Generate data in batches
batch<-generator_next(train_gen)

#Data Augmentation
gen_flip=image_data_generator(vertical_flip = TRUE,horizontal_flip = TRUE)
train_gen_flip=flow_images_from_directory(paste0(base_dir,"\\train"),generator = gen_flip,batch_size=32,
                                     save_to_dir=paste0(base_dir,"\\augumented_images"),shuffle = FALSE,
                                     target_size = c(224,224))
#Generate data in batches
batch<-generator_next(train_gen_flip)
