from training import Knn, NeuralNetwork


knn = Knn()
knn.train('new-features.txt')
#knn.externalValidateModel('feature-single-individual.txt')
knn.splitValidateModel()
knn.crossValidateModel()
#knn.visualizeData('new-features.txt')
knn.visualizePredictedDataset()



#nn = NeuralNetwork()
#nn.trainMLP('feature-data.txt')
