import operator
import csv
import math
import random

def loadDataset(filename, split, trainingSet=[] , testSet=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)-1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])

def EuclidDist(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)

def getNeighbors(trainSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	#print("length = ",length)
	for x in range(len(trainSet)):
		dist = EuclidDist(testInstance, trainSet[x], length)
		distances.append((trainSet[x], dist))
	#print(distances)
	#distances.sort(key=operator.itemgetter(1))
	distances.sort(key=lambda distances : distances [1])
	#print(distances)
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
		#print(distances[x])
	return neighbors

def getClass(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		instance_class = neighbors[x][-1]
		if instance_class in classVotes:
			classVotes[instance_class] += 1
		else:
			classVotes[instance_class] = 1
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0
    
    
def main():
	# prepare data
	trainingSet=[]
	testSet=[]
	split = 0.7
	loadDataset('iris.data', split, trainingSet, testSet)
	print ('训练集合: ' + repr(len(trainingSet)))
	print ('测试集合: ' + repr(len(testSet)))
	# generate predictions
	predictions=[]
	k = 3
	for x in range(len(testSet)):
		neighbors = getNeighbors(trainingSet, testSet[x], k)
		result = getClass(neighbors)
		predictions.append(result)
		print('> 预测=' + repr(result) + ', 实际=' + repr(testSet[x][-1]))
	accuracy = getAccuracy(testSet, predictions)
	print('精确度为: ' + repr(accuracy) + '%')
	
main()
