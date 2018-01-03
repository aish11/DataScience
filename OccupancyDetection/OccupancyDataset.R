#########OCCUPANCY DETECTION##################
#######CODE BY: AISHWARYA SRIVASTAVA##########

rm(list=ls())
data_train<-read.csv("C:\\Users\\Aishwarya\\Downloads\\DS\\occupancy_data\\datatraining.txt")
data_test1<-read.csv("C:\\Users\\Aishwarya\\Downloads\\DS\\occupancy_data\\datatest.txt")
data_test2<-read.csv("C:\\Users\\Aishwarya\\Downloads\\DS\\occupancy_data\\datatest2.txt")

head(data_train)
dim(data_train)
dim(data_test1)
dim(data_test2)
#Check if data is clean and doen not contain any null value
sum(is.na(data_train))
sum(is.na(data_test1))
sum(is.na(data_test2))

#Convert the date column in a proper format
data_train$date <- (as.Date(data_train$date))
data_test1$date <- as.Date(data_test1$date)
data_test2$date <- as.Date(data_test2$date)
data_test<- rbind(data_test1,data_test2)

#Exploratory Data Analysis 
library(ggplot2)
Complete_data <- rbind(data_train,data_test,data_test1)

##Occupancy variation with temperature
ggplot(Complete_data, aes(Complete_data$Temperature,fill = factor(Occupancy)))+
  geom_histogram(alpha = 0.7,bins = 100,
                 position = 'identity')+
  xlab("Temperature") +
  ylab("Frequency") +
  ggtitle("Occupancy variation with Temperature")

##Occupancy variation with date
ggplot(Complete_data, aes(Complete_data$date, fill = factor(Occupancy))) +
  geom_density(alpha = 0.5) + xlab("Date") +
  ylab("Frequency") +
  ggtitle("Occupancy variation with Date")


##Occupancy variation with light
ggplot(Complete_data, aes(Complete_data$Light, fill = factor(Occupancy))) +
  geom_density(alpha = 0.5)+
  xlab("Light") +
  ylab("Frequency") +
  xlim(0,1698)+
  ggtitle("Occupancy variation with Light")


##Occupancy variation with Humidity

ggplot(Complete_data, aes(Complete_data$Humidity,fill = factor(Occupancy)))+
  geom_histogram(alpha = 0.7,bins = 100,
                 position = 'identity')+
  xlab("Humidity") +
  ylab("Frequency") +
  ggtitle("Occupancy variation with Humidity")

##Occupancy variation with Humidity Ratio
ggplot(Complete_data, aes(Complete_data$HumidityRatio,fill = factor(Occupancy)))+
  geom_histogram(alpha = 0.7,bins = 100,
                 position = 'identity')+
  xlab("HumidityRatio") +
  ylab("Frequency") +
  ggtitle("Occupancy variation with HumidityRatio")

##Occupancy variation with CO2

ggplot(Complete_data, aes(Complete_data$CO2, fill = factor(Occupancy))) +
  geom_density(alpha = 0.5) + xlab("CO2") +
  ylab("Frequency") +
  ggtitle("Occupancy variation with CO2")

##VARIABLE SELECTION
library(leaps)
library(MASS)
library(caret)

#Stepwise AIC
full=glm(Occupancy ~ ., data=data_train)
summary(full)
step = stepAIC(full,trace=FALSE)
step$anova

#Best subset selection
regfit.full = regsubsets(Occupancy ~ ., data=data_train)
summary(regfit.full)
plot(regfit.full, scale="adjr2")
plot(regfit.full, scale="bic")


#MODEL BUILDING

library(DBI)
library(biglm)
library(e1071)
library(caret)
head(data_train)

data_train <- subset(data_train,select = -c(Humidity))
data_test <- subset(data_test,select = -c(Humidity))

data_train$Occupancy<-as.factor(data_train$Occupancy)
data_test$Occupancy<-as.factor(data_test$Occupancy)


##LOGISTIC REGRESSION
log_reg <- train( Occupancy ~ ., data=data_train, method = "glm",family = "binomial",trControl=trainControl(method = "cv", number = 5)) 
log_reg_predict <- predict(log_reg, data_test)
confusion_matrix_reg <- confusionMatrix(log_reg_predict, data_test$Occupancy)
confusion_matrix_reg

confusion_matrix_reg <- as.data.frame(as.table(confusion_matrix_reg))
plot1 <- ggplot(confusion_matrix_reg)
plot1 + geom_tile(aes(x=Prediction, y=Reference, fill=Freq)) + 
  scale_x_discrete(name="Actual Class") + scale_y_discrete(name="Predicted Class") + 
  scale_color_gradient(breaks=seq(from=0, to=2000000, by=500)) + labs(fill="Frequency")

##Random forest
library(randomForest)
random_forest1 <- randomForest(data_train, data_train$Occupancy, proximity = TRUE, importance = TRUE,ntree=2)

random_forest1 <- randomForest(data_train, data_train$Occupancy)

random_forest_validation1 <- predict(random_forest1, data_test)
confusion_matrix_forest <- confusionMatrix(random_forest_validation1, data_test$Occupancy)
confusion_matrix_forest

confusion_matrix_forest <- as.data.frame(as.table(confusion_matrix_forest))
plot1 <- ggplot(confusion_matrix_forest)
plot1 + geom_tile(aes(x=Prediction, y=Reference, fill=Freq)) + 
  scale_x_discrete(name="Actual Class") + scale_y_discrete(name="Predicted Class") + 
  scale_color_gradient(breaks=seq(from=0, to=2000000, by=500)) + labs(fill="Frequency")

##KNN
knn <- trainControl(method = "repeatedcv", number = 10,repeats = 3)
Knn_fit <- train(Occupancy~., data =data_train,
                 method = "knn",
                 tuneGrid = data.frame(k = 10),
                 preProcess = c("scale","center"))
result = predict(Knn_fit,data_test)
confusionmatrixKNN <- confusionMatrix(data_test$Occupancy,result,positive="1")
confusionmatrixKNN

confusionmatrixKNN <- as.data.frame(as.table(confusionmatrixKNN))
plot1 <- ggplot(confusionmatrixKNN)
plot1 + geom_tile(aes(x=Prediction, y=Reference, fill=Freq)) + 
  scale_x_discrete(name="Actual Class") + scale_y_discrete(name="Predicted Class") + 
  scale_color_gradient(breaks=seq(from=0, to=2000000, by=500)) + labs(fill="Frequency")


######################################################