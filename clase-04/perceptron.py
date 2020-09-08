import random
#Basic version of Perceptron that builds from one parameter (size of the input of the perceptron) and initilizes its parameters randomly
#it also has a simple learning and training methods
class Perceptron:
    def __init__(self, input_size, lr=0.1):
        self.input_size = input_size
        self.w = []
        self.lr = lr
        for i in range(input_size):
            self.w += [4*random.random() - 2]
        self.b = 4*random.random() - 2
    
    def forward(self, x):
        result = sum([w*i for (w,i) in zip(self.w,x)]) + self.b
        return 1 if result > 0 else 0

    def learn(self, data, output, target):
        diff = target - output
        for (i, d) in zip(range(self.input_size), data):
            self.w[i] += self.lr*d*diff
        self.b += self.lr*diff

    def train(self, dataset, target):
        for (d, t) in zip(dataset, target):
            output = self.forward(d)
            self.learn(d, output, t)

            
#test: model training to label an (x,y) input to 1 or 0 depending if it is on top or below of f(x)=x curve between [(-50,-50), (50,50)]
#training data creation
dataset = []
target = []
for i in range(50):
    x = 100*random.random() - 50
    y = 100*random.random() - 50
    t = 1 if y >= x else 0
    dataset += [(x,y)]
    target += [t]
#perceptron training
p = Perceptron(2)
p.train(dataset, target)
#testing data creation
dataset = []
target = []
for i in range(100):
    x = 100*random.random() - 50
    y = 100*random.random() - 50
    t = 1 if y >= x else 0
    dataset += [(x,y)]
    target += [t]
#evaluating trained model performance on testing data
correct = []
for (d, t) in zip(dataset, target):
    output = p.forward(d)
    if output == t:
        correct += [output]
print("Model accuracy: ", len(correct)/len(target))
