#Basic version of Perceptron that only initializes it from a weights list and have forward method
class Perceptron:
    def __init__(self, w, b = 0):
        self.size_input = len(w)
        self.w = w
        if w is None:
            self.w = [1] * self.size_input   
        self.b = b
    
    def forward(self, x):
        result = sum([w*i for (w,i) in zip(self.w,x)]) + self.b
        return 1 if result > 0 else 0
    

#testing or Perceptron
p = Perceptron([1,1], -0.5)
assert p.forward([0,0]) == 0
assert p.forward([1,1]) == 1
assert p.forward([1,0]) == 1
assert p.forward([0,1]) == 1
