"""
obsolete plotting function code
"""
import csv
import matplotlib.pyplot as plt
import numpy as np
accuracies = []
nmis = []
balance = []
entropy = []
cases = ["BOTH 0.40","BOTH 0.10","USPS 0.40","USPS 0.10","MNIST 0.40","MNIST 0.10" ]
with open('results.txt') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=' ')
    i = 0
    for row in csv_reader:
        if i%5 == 0:
            accuracies += [[float(i) for i in row]]
        elif i%5 == 1:
            nmis += [[float(i) for i in row]]
        elif i%5 == 2:
            balance += [[float(i) for i in row]]
        elif i%5 == 3:
            temp_entropy = [float(i) for i in row]
        elif i%5 == 4:
            entropy += [np.array([float(i) for i in row])/np.array(temp_entropy)]
            
        i += 1

def plotter(plotlist,name):        
    for i in range(len(plotlist)):
        plt.plot(plotlist[i],label=cases[i])
    plt.title(str(name) + " per category of corruption")
    plt.legend()  
    plt.show()
plotter(accuracies,"accuracies")
plotter(nmis,"nmis")

plotter(balance,"balance")
plotter(entropy,"entropy")
