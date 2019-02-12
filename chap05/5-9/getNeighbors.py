import operator
import math

def EuclidDist(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)

def getNeighbors(trainSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainSet)):
		dist = EuclidDist(testInstance, trainSet[x], length)
		distances.append((trainSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

trainSet = [[3, 2, 6, 'a'], [1, 2, 4, 'b'],[2, 2, 2, 'b'],[1, 5, 4, 'a']]
testInstance = [4, 6, 7]
k = 1
neighbors = getNeighbors(trainSet, testInstance, 1)
print(neighbors)