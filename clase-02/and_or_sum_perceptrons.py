from perceptron import Perceptron

#Class for circuit built from perceptrons that sums bits
class Add_Circuit:
    def __init__(self):
        create_nand_gate = lambda: Perceptron([-2,-2], 3)

        self.g1 = create_nand_gate()
        self.g2 = create_nand_gate()
        self.g3 = create_nand_gate()
        self.g4 = create_nand_gate()
        self.g5 = create_nand_gate()

    def sum(self, x1, x2):
        out_g_1 = self.g1.forward([x1,x2])
        out_g_2 = self.g2.forward([x1, out_g_1])
        out_g_3 = self.g3.forward([out_g_1, x2])
        out_g_4 = self.g4.forward([out_g_2, out_g_3])
        out_g_5 = self.g5.forward([out_g_1, out_g_1])

        return out_g_5*10 + out_g_4 


#Create logic gates from Perceptrons chosing its weights and bias before
or_perceptron = Perceptron([1,1], -0.5)
and_perceptron = Perceptron([1,1], -1.5)
nand_perceptron = Perceptron([-2,-2], 3)

#Create add bit circuit
add_circuit = Add_Circuit()



#testing or perceptron
assert or_perceptron.forward([0,1]) == 1
assert or_perceptron.forward([0,0]) == 0
#testing and perceptron
assert and_perceptron.forward([0,0]) == 0
assert and_perceptron.forward([1,1]) == 1
#testing nand perceptron
assert nand_perceptron.forward([0,0]) == 1
assert nand_perceptron.forward([1,1]) == 0
#testing add_circuit
assert add_circuit.sum(0,0) == 0
assert add_circuit.sum(0,1) == 1
assert add_circuit.sum(1,1) == 10
