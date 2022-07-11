import numpy as np

class Optimizers:

    def __init__(self, num_weight, learning_rate):
        self.m = [0] * num_weight
        self.v = [0] * num_weight
        self.t = 1
        self.learning_rate = learning_rate
        
    def Adam(self, params, grads, beta1 = 0.9,beta2 = 0.999):
        """ Adam optimizer, bias correction is implemented. """

        params = params.flatten()
        grads = grads.flatten()

        updated_params = []
        
        for  i, (param, grad) in enumerate(zip(params, grads)):
          self.m[i] = beta1 * self.m[i] + (1-beta1) * grad          
          self.v[i] = beta2 * self.v[i] + (1-beta2) * grad **2
          m_corrected = self.m[i] / (1-beta1**self.t)
          v_corrected = self.v[i] / (1-beta2**self.t)
          param = param - (self.learning_rate * m_corrected / (np.sqrt(v_corrected) + 1e-8))
          updated_params.append(
            param 
          )
          
        self.t +=1
        return updated_params