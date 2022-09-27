import torch
import numpy as np
class MLP:
    def __init__(
        self,
        linear_1_in_features,
        linear_1_out_features,
        f_function,
        linear_2_in_features,
        linear_2_out_features,
        g_function
    ):
        """
        Args:
            linear_1_in_features: the in features of first linear layer
            linear_1_out_features: the out features of first linear layer
            linear_2_in_features: the in features of second linear layer
            linear_2_out_features: the out features of second linear layer
            f_function: string for the f function: relu | sigmoid | identity
            g_function: string for the g function: relu | sigmoid | identity
        """
        self.f_function = f_function
        self.g_function = g_function

        self.parameters = dict(
            W1 = torch.randn(linear_1_out_features, linear_1_in_features),
            b1 = torch.randn(linear_1_out_features),
            W2 = torch.randn(linear_2_out_features, linear_2_in_features),
            b2 = torch.randn(linear_2_out_features),
        )
        self.grads = dict(
            dJdW1 = torch.zeros(linear_1_out_features, linear_1_in_features),
            dJdb1 = torch.zeros(linear_1_out_features),
            dJdW2 = torch.zeros(linear_2_out_features, linear_2_in_features),
            dJdb2 = torch.zeros(linear_2_out_features),
        )

        # put all the cache value you need in self.cache
        self.cache = {}
        

    def forward(self, x):
        """
        Args:
            x: tensor shape (batch_size, linear_1_in_features)
        """

        self.cache['nonlinear_funcs'] = {'relu':lambda x:(x>0)*x,
                 'sigmoid': lambda x: (1+exp(-x))**-1,
                 'identity':lambda x:x 
        }
        W1,b1 = self.parameters['W1'], self.parameters['b1']
        z1 = torch.matmul(x,W1.T) +b1
        print(z1.size())
        f_function = self.cache['nonlinear_funcs'][self.f_function]
        z2 = z1.apply_(f_function)
        print(z2.size())
        W2,b2 = self.parameters['W2'], self.parameters['b2']
        z3 = torch.matmul(z2,W2.T)+b2
        print(z3.size())
        g_function = self.cache['nonlinear_funcs'][self.g_function]
      

        y_hat = z3.apply_(g_function)

        #getting Jacobian of W1 w.r.t z1
        W_rows = self.parameters['W1'].size()[1]
        Jacobian = np.einsum('ik, jk', np.eye(W_rows, W_rows),x)
        print('Jacobian size:', Jacobian.shape)
        self.cache['computation_graph'] = ['g_function', 'W2','b2','f_function','W1','b1']
        self.cache['values'] = {'z3':z3,
                                'z2':z2,
                                'z1':z1,
                                'Jacobian': torch.from_numpy(Jacobian)}
        return(y_hat)
    
    def backward(self, dJdy_hat):
        """
        Args:
            dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
        """ 
        self.cache['nonlinear_grads'] = {'relu':lambda x:(x>0)*1,
                 'sigmoid': lambda x: (1+exp(-x))**-1*(1-(1+exp(-x))**-1), 
                 'identity':'identity'
        }
        grad_out = dJdy_hat # (10x5)
        while self.cache['computation_graph']:
            val = self.cache['computation_graph'].pop(0)

            if val == 'g_function':
                deriv = self.cache['nonlinear_grads'][self.g_function]
                if deriv == 'identity':
                    grad_out = grad_out
                    print(1,grad_out.size()) # (10x5, element wise operation)
                else:
                    grad_out = grad_out.apply_(deriv)
                    print(1,grad_out.size())
            elif val == 'W2':
                grad_out = torch.matmul(self.cache['values']['z2'].T,grad_out)
                #20x10*10x5 = 20x5
                self.grads['dJdW2'] = grad_out
                print(2,grad_out.size())
            elif val == 'b2':
                self.grads['dJb2'] = grad_out
                print(3,grad_out.size()) #20x5
            elif val == 'f_function':
                deriv = self.cache['nonlinear_grads'][self.f_function]
                if deriv == 'identity':
                    grad_out = grad_out
                    print(4,grad_out.size()) #20x5
                else:
                    grad_out = grad_out.apply_(deriv)
                    print(5,grad_out.size()) #20x5
            elif val == 'W1':
                #W1 is 2x20
                #2x20*20x5  = 2x20
                self.grads['dJdW1'] = torch.matmul(self.cache['values']['Jacobian'].T,grad_out)
            elif val == 'b1':
                self.grads['dJb2'] = grad_out
                print(6,grad_out.size())
        self.parameters['W2'] -= self.grads['dJdW2']
        self.parameters['b2'] -= self.grads['dJdb2']
        self.parameters['W1'] -= self.grads['dJdW1']
        self.parameters['b2'] -= self.grads['dJdb1']

    
    def clear_grad_and_cache(self):
        for grad in self.grads:
            self.grads[grad].zero_()
        self.cache = dict()

def mse_loss(y, y_hat):
    """
    Args:
        y: the label tensor (batch_size, linear_2_out_features)
        y_hat: the prediction tensor (batch_size, linear_2_out_features)

    Return:
        J: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    loss = (y-y_hat)**2
    dJdy_hat = 2*(y_hat-y)

    return(loss, dJdy_hat)

def bce_loss(y, y_hat):
    """
    Args:
        y_hat: the prediction tensor
        y: the label tensor
        
    Return:
        loss: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # TODO: Implement the bce loss
    pass

    # return loss, dJdy_hat











