import DTree
import pandas as pd

D2 = pd.read_csv('data/D2.txt', sep=' ', names=['X1','X2','Y'])

model = DTree.DecisionTreeClassifier()
tree = model.train(D2, ['X1','X2'], 'Y')

model.print_tree()
