import torch, pandas as pd, numpy as np
from torch.autograd import Variable

df_train = pd.read_csv("train.csv")
print(df_train)
x_data = np.array(df_train[['ed', 'inc', 'pop']])
y_data = np.array(df_train[['cases']])

x_data = Variable(torch.Tensor(x_data.tolist())) 
y_data = Variable(torch.Tensor(y_data.tolist())) 

class LinearRegressionModel(torch.nn.Module): 
  
    def __init__(self): 
        super(LinearRegressionModel, self).__init__() 
        nodeCts = [3, 6, 4, 2, 1, 1]
        netSpec = [torch.nn.Sigmoid() if i%2 else torch.nn.Linear(nodeCts[i//2], nodeCts[1+i//2]) for i in range(2*len(nodeCts)-2)]
        self.linear = torch.nn.Sequential(*netSpec)
  
    def forward(self, x): 
        y_pred = self.linear(x) 
        return y_pred 

our_model = LinearRegressionModel() 
  
criterion = torch.nn.MSELoss() 
optimizer = torch.optim.SGD(our_model.parameters(), lr = 0.01) 
  
for epoch in range(20): 

    pred_y = our_model(x_data) 
  
    loss = criterion(pred_y, y_data) 
  
    optimizer.zero_grad() 
    loss.backward() 
    optimizer.step() 
    print('epoch {}, loss {}'.format(epoch, loss.item())) 

df_test = pd.read_csv("test.csv")
pred_x = np.array(df_test[['ed', 'inc', 'pop']])
new_var = Variable(torch.Tensor(pred_x.tolist())) 
pred_y = our_model(new_var)
print(pred_y)