from training import Knn, NeuralNetwork

#knn = Knn()
#knn.train('new-features.txt')
#knn.externalValidateModel('feature-single-individual.txt')
#knn.splitValidateModel()
#knn.crossValidateModel()
#knn.visualizeData('new-features.txt')
#knn.visualizePredictedDataset()

nn = NeuralNetwork()
#nn.trainMLP('feature-data.txt')
#nn.trainSoftmax('feature-data.txt')
nn.trainLimitedSoftmax('feature-data.txt', 50)
#nn.trainLimitedMLP('feature-data.txt', 50)


#knn.trainLimited('feature-data.txt', 50)