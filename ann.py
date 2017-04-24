from features import getFeatures
from random import randint
from network import NeuralNet
import numpy as np
    
target_map = {

    'automobile':       [1, 0, 0],
    'frog':             [0, 1, 0],
    'airplane':         [0, 0, 1],
}

fp = open('trainLabels.csv')

input = []
input_id = []
target = []

classes = ['automobile', 'frog', 'airplane']

for x in xrange(1, 1000):
    line = fp.readline()
    strs = line.strip('\n').split(',')
    if strs[1] in classes:
        img_path = './train/' + strs[0] + '.png'
        input.append(getFeatures(img_path))
        input_id.append(strs[0] + '.png')
        target.append(target_map[strs[1]])

input_normal = [[[0,0] for c in range(len(input[0]))] for r in range(len(input))]

for x in range(len(input)):
    mx = max(input[x])
    for y in range(len(input[0])):
        input_normal[x][y] = input[x][y]/mx

n = NeuralNet(len(input[0]), 102, 102, len(target[0]))

n.train(input_normal, target)

predicted = []

testing = []

testing_id = []

target_test = []

for x in xrange(1, 1000):
    line = fp.readline()
    strs = line.strip('\n').split(',')
    if strs[1] in classes:
        img_path = './train/' + strs[0] + '.png'
        testing_id.append(strs[0])
        testing.append(getFeatures(img_path))
        target_test.append(target_map[strs[1]])

testing_normal = [[[0,0] for c in range(len(testing[0]))] for r in range(len(testing))]

for x in range(len(testing)):
    mx = max(testing[x])
    for y in range(len(testing[0])):
        testing_normal[x][y] = testing[x][y]/mx


for x in range(len(testing)):
    p = network.predict([testing_normal[x]])
    predicted.append(p)
    print str(testing_id[x]) + ' => ' + str(p) + ' ' + str(target_test[x])