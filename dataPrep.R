# Matthew L. Miller
# mmiller319

# Read in mnist_train.csv and mnist_test.csv
train <- read.csv(file="mnist_train.csv", header=FALSE, sep=",")
test <- read.csv(file="mnist_test.csv", header = FALSE, sep = ",")

# Partition the training set for 0,1 and 3,5
train_data_0_1 <- train[ which(train[785,] == 0 | train[785,] == 1)]
train_data_3_5 <- train[ which(train[785,] == 3 | train[785,] == 5)]
test_data_0_1 <- test[ which(test[785,] == 0 | test[785,] == 1)]
test_data_3_5 <- test[ which(test[785,] == 3 | test[785,] == 5)]

# Separate class labels (row 785)
test_labels_0_1 <- test_data_0_1[785,]
test_labels_3_5 <- test_data_3_5[785,]
train_labels_0_1 <- train_data_0_1[785,]
train_labels_3_5 <- train_data_3_5[785,]

# Remove row 785 from image data
test_data_0_1 <- test_data_0_1[-c(785), ]
test_data_3_5 <- test_data_3_5[-c(785), ]
train_data_0_1 <- train_data_0_1[-c(785), ]
train_data_3_5 <- train_data_3_5[-c(785), ]

# Get image vectors
image0 <- train_data_0_1[,1]
image1 <- train_data_0_1[,12665]
image3 <- train_data_3_5[,1]
image5 <- train_data_3_5[,11552]

# Convert image data to a matrix
matrix0 <- matrix(image0, nrow=28, ncol=28, byrow=FALSE)
matrix1 <- matrix(image1, nrow=28, ncol=28, byrow=FALSE)
matrix3 <- matrix(image3, nrow=28, ncol=28, byrow=FALSE)
matrix5 <- matrix(image5, nrow=28, ncol=28, byrow=FALSE)

# Create rotation function to correct for image() function's rotation
# Taken from: https://www.r-bloggers.com/creating-an-image-of-a-matrix-in-r-using-image/
rotate <- function(x) t(apply(x, 2, rev))

# Display images in greyscale using rotate function
image(rotate(matrix0), col=gray(0:255/255))
image(rotate(matrix1), col=gray(0:255/255))
image(rotate(matrix3), col=gray(0:255/255))
image(rotate(matrix5), col=gray(0:255/255))