class Scaler:
    def __init__(self):
        self.mean = 0
        self.std = 1
    
    def fit(self, data):
        self.mean = data.mean(axis=0, keepdims=True)
        self.std = data.std(axis=0, keepdims=True)
        
    def transform(self, data):
        return (data - self.mean)/self.std
    
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

def one_hot(array, num_classes):
    ret = np.zeros((len(array), num_classes))
    for i, val in enumerate(array):
        ret[i, val] = 1
    return ret.astype(int)

def accuracy(y_true, y_pred):
    ret = np.zeros(y_true.shape[1])
    for i in range(y_true.shape[1]):
        if (y_true[:, i] == y_pred[:, i]).all():
            ret[i] = 1
    return ret.sum()/ret.shape[0]

def train_test_split(X, y=None, train_size=0.7, shuffle=True, random_state=None):

    np.random.seed(random_state)

    permutation = list(np.random.permutation(X.shape[0]))
    border = round(train_size*len(X))
    if y is None:
        return X[permutation][:border], X[permutation][border:]
    else:
        return X[permutation][:border], y[permutation][:border], X[permutation][border:], y[permutation][border:]

class MLP:
    
    def __init__(self, layers_dims=None, activations=None, optimizer="gd", learning_rate=0.005,
                 mini_batch_size=64, beta1=0.9, beta2=0.999, epsilon=1e-8, random_state=None):
        np.random.seed(random_state)
        self.hyperparam_dict = {"layers_dims": layers_dims, 
                                "activations": activations, 
                                "optimizer": optimizer,
                                "learning_rate": learning_rate,
                                "mini_batch_size": mini_batch_size,
                                "beta1": beta1,
                                "beta2": beta2,
                                "epsilon": epsilon,
                                "random_state": random_state}

    def sigmoid(self, Z):
        return 1/(1+np.exp(-Z)), Z
    
    
    def sigmoid_backward(self, dA, Z):
        s, _ = self.sigmoid(Z)
        return dA * s * (1 - s)
    
    
    def relu(self, Z):
        return np.maximum(0, Z), Z
    
    
    def relu_backward(self, dA, Z):
        dZ = np.array(dA, copy=True)
        dZ[Z<=0] = 0
        return dZ
    
    
    def initialize_parameters(self, layers_dims):
        assert len(layers_dims)-1 == len(self.hyperparam_dict["activations"])

        self.parameters = {}
        L = len(layers_dims)
        for l in range(1, L):
            self.parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * np.sqrt(1/layers_dims[l-1])
            self.parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))
            
            assert(self.parameters['W' + str(l)].shape == (layers_dims[l], layers_dims[l-1]))
            assert(self.parameters['b' + str(l)].shape == (layers_dims[l], 1))
    
    def random_mini_batches(self, X, Y, mini_batch_size, random_state=None):
        np.random.seed(random_state)
        m = X.shape[1]
        mini_batches = []
        permutation = list(np.random.permutation(m))
        X_shuffled = X[:, permutation]
        Y_shuffled = Y[:, permutation].reshape((-1, m))
        num_complete_mini_batches = np.floor_divide(m, mini_batch_size)
        for t in range(num_complete_mini_batches):
            mini_batch_X = X_shuffled[:, t*mini_batch_size:(t+1)*mini_batch_size]
            mini_batch_Y = Y_shuffled[:, t*mini_batch_size:(t+1)*mini_batch_size]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        if m % mini_batch_size != 0:
            last_mini_batch_X = X_shuffled[:, num_complete_mini_batches*mini_batch_size:]
            last_mini_batch_Y = Y_shuffled[:, num_complete_mini_batches*mini_batch_size:]
            mini_batch = (last_mini_batch_X, last_mini_batch_Y)
            mini_batches.append(mini_batch)
            
        return mini_batches
    
    def linear_forward(self, A, W, b):
        
        Z = W.dot(A) + b
        cache = (A, W, b)
        
        return Z, cache
    
    
    def linear_activation_forward(self, A_prev, W, b, activation):
        
        if activation=="sigmoid":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.sigmoid(Z)
        elif activation=="relu":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.relu(Z)
            
        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        cache = (linear_cache, activation_cache)
        
        return A, cache
    
    
    def forward(self, X):
        
        caches = []
        A = X
        L = len(self.parameters) // 2
        for l in range(0, L):
            A_prev = A
            A, cache = self.linear_activation_forward(A_prev,
                                                      self.parameters["W"+str(l+1)],
                                                      self.parameters["b"+str(l+1)],
                                                      self.hyperparam_dict["activations"][l])
            caches.append(cache)
        assert(A.shape == (self.parameters["W"+str(L)].shape[0], X.shape[1]))
        return A, caches
    
    
    def compute_cost(self, AL, Y):
        m = Y.shape[0]
        eps=1e-10
        AL = np.clip(AL, eps, 1-eps)
        Y = Y.reshape(AL.shape)
        return -1 / m * np.sum( Y*np.log(AL) + (1-Y)*np.log(1-AL))
    
    
    def linear_backward(self, dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]
        dW = 1/m * np.dot(dZ, A_prev.T)
        db = 1/m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
        
        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)
        
        return dA_prev, dW, db
    
    
    def linear_activation_backward(self, dA, cache, activation):
        
        linear_cache, activation_cache = cache
        if activation == "relu":
            dZ = self.relu_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
        elif activation == "sigmoid":
            dZ = self.sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
        return dA_prev, dW, db
    
    
    def backward(self, AL, Y, caches, activations):
        
        grads = {}
        L = len(caches)
        Y = Y.reshape(AL.shape)
        eps=1e-10
        AL = np.clip(AL, eps, 1-eps)
        dA = -np.divide(Y, AL) + np.divide(1 - Y, 1 - AL)
        current_cache = caches[L-1]
        
        grads["dA"+str(L-1)], grads["dW"+str(L)], grads["db"+str(L)] =\
                self.linear_activation_backward(dA, current_cache, activations[L-1])
        if L > 1:
            for l in reversed(range(L-1)):
                current_cache = caches[l]
                dA_temp, dW_temp, db_temp = self.linear_activation_backward(grads["dA"+str(l+1)],
                                                                            current_cache, activations[l])
                grads["dA"+str(l)] = dA_temp
                grads["dW"+str(l+1)] = dW_temp
                grads["db"+str(l+1)] = db_temp
        return grads
    
    def initialize_velocity(self):
        L = len(self.parameters) // 2
        v = {}
        for l in range(L):
            v["dW" + str(l+1)] = np.zeros_like(self.parameters["W" + str(l+1)])
            v["db" + str(l+1)] = np.zeros_like(self.parameters["b" + str(l+1)])
        return v
    
    def initialize_rmsprop(self):
        L = len(self.parameters) // 2
        print(L)
        s = {}
        for l in range(L):
            s["dW" + str(l+1)] = np.zeros_like(self.parameters["W" + str(l+1)])
            s["db" + str(l+1)] = np.zeros_like(self.parameters["b" + str(l+1)])
        return s
    
    def initialize_adam(self):
        L = len(self.parameters) // 2
        v = {}
        s = {}
        for l in range(L):
            v["dW" + str(l+1)] = np.zeros_like(self.parameters["W" + str(l+1)])
            v["db" + str(l+1)] = np.zeros_like(self.parameters["b" + str(l+1)])
            s["dW" + str(l+1)] = np.zeros_like(self.parameters["W" + str(l+1)])
            s["db" + str(l+1)] = np.zeros_like(self.parameters["b" + str(l+1)])
        return v, s
    
    def update_parameters_with_gd(self, grads, learning_rate):
        L = len(self.parameters) // 2
        for l in range(L):
            self.parameters["W"+str(l+1)] = self.parameters["W"+str(l+1)] - learning_rate*grads["dW"+str(l+1)]
            self.parameters["b"+str(l+1)] = self.parameters["b"+str(l+1)] - learning_rate*grads["db"+str(l+1)]
    
    def update_parameters_with_momentum(self, grads, v, beta1=0.9, learning_rate=0.01):
        L = len(self.parameters) // 2
        for l in range(L):
            v["dW" + str(l+1)] = beta1 * v["dW" + str(l+1)] + (1 - beta1) * grads["dW" + str(l+1)]
            v["db" + str(l+1)] = beta1 * v["db" + str(l+1)] + (1 - beta1) * grads["db" + str(l+1)]
            self.parameters["W" + str(l+1)] = self.parameters["W" + str(l+1)] - learning_rate * v["dW" + str(l+1)]
            self.parameters["b" + str(l+1)] = self.parameters["b" + str(l+1)] - learning_rate * v["db" + str(l+1)]
        return v
    
    def update_parameters_with_rmsprop(self, grads, s, beta2=0.999, learning_rate=0.01, epsilon=10e-8):
        L = len(self.parameters) // 2
        for l in range(L):
            s["dW" + str(l+1)] = beta2 * s["dW" + str(l+1)] + (1 - beta2) * grads["dW" + str(l+1)]**2
            s["db" + str(l+1)] = beta2 * s["db" + str(l+1)] + (1 - beta2) * grads["db" + str(l+1)]**2
            self.parameters["W" + str(l+1)] = self.parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)] / (np.sqrt(s["dW" + str(l+1)]) + epsilon)
            self.parameters["b" + str(l+1)] = self.parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)] / (np.sqrt(s["db" + str(l+1)]) + epsilon)
        return s
    
    def update_parameters_with_adam(self, grads, v, s, t, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon = 10e-8):
        L = len(self.parameters) // 2
        v_corrected = {}
        s_corrected = {}

        for l in range(L):
            v["dW" + str(l+1)] = beta1 * v["dW" + str(l+1)] + (1 - beta1) * grads["dW" + str(l + 1)]
            v["db" + str(l+1)] = beta1 * v["db" + str(l+1)] + (1 - beta1) * grads["db" + str(l + 1)]
            v_corrected["dW" + str(l+1)] = v["dW" + str(l+1)] / (1 - beta1**t)
            v_corrected["db" + str(l+1)] = v["db" + str(l+1)] / (1 - beta1**t)
            s["dW" + str(l+1)] = beta2 * s["dW" + str(l+1)] + (1 - beta2) * grads["dW" + str(l + 1)]**2
            s["db" + str(l+1)] = beta2 * s["db" + str(l+1)] + (1 - beta2) * grads["db" + str(l + 1)]**2
            s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)] / (1 - beta2**t)
            s_corrected["db" + str(l+1)] = s["db" + str(l+1)] / (1 - beta2**t)
            self.parameters["W" + str(l+1)] = self.parameters["W" + str(l+1)] - learning_rate * v_corrected["dW" + str(l+1)] / (np.sqrt(s_corrected["dW" + str(l+1)]) + epsilon)
            self.parameters["b" + str(l+1)] = self.parameters["b" + str(l+1)] - learning_rate * v_corrected["db" + str(l+1)] / (np.sqrt(s_corrected["db" + str(l+1)]) + epsilon)
        return v, s
        
    def fit(self, X_train, y_train, num_iterations=2000, print_cost=False, visualize=False):
        
        eps=1e-2
        costs=[]
        self.initialize_parameters(self.hyperparam_dict["layers_dims"])
        seed = self.hyperparam_dict["random_state"]
        if self.hyperparam_dict["optimizer"] == "gd":
            pass
        elif self.hyperparam_dict["optimizer"] == "momentum":
            v = self.initialize_velocity()
        elif self.hyperparam_dict["optimizer"] == "rmsprop":
            s = self.initialize_rmsprop()
        elif self.hyperparam_dict["optimizer"] == "adam":
            v, s = self.initialize_adam()
            t = 0
        
        iterator = tqdm_notebook if visualize else iter
        for i in iterator(range(0, num_iterations)):
            
            seed = seed + 1
            minibatches = self.random_mini_batches(X_train, y_train, self.hyperparam_dict["mini_batch_size"], seed)
            for minibatch in minibatches:
                minibatch_X, minibatch_Y = minibatch
                A, caches = self.forward(minibatch_X)
                cost = self.compute_cost(A, minibatch_Y)
                grads = self.backward(A, minibatch_Y, caches, self.hyperparam_dict["activations"])
                
                if self.hyperparam_dict["optimizer"] == "gd":
                    self.update_parameters_with_gd(grads, self.hyperparam_dict["learning_rate"])
                
                elif self.hyperparam_dict["optimizer"] == "momentum":
                    v = self.update_parameters_with_momentum(grads, v, self.hyperparam_dict["beta1"],
                
                                                             self.hyperparam_dict["learning_rate"])
                elif self.hyperparam_dict["optimizer"] == "rmsprop":
                    s = self.update_parameters_with_rmsprop(grads, s, self.hyperparam_dict["beta2"],
                                    self.hyperparam_dict["learning_rate"], self.hyperparam_dict["epsilon"])
                
                elif self.hyperparam_dict["optimizer"] == "adam":
                    t += 1
                    v, s = self.update_parameters_with_adam(grads, v, s, t,
                                     self.hyperparam_dict["learning_rate"], self.hyperparam_dict["beta1"],
                                     self.hyperparam_dict["beta2"], self.hyperparam_dict["epsilon"])
            
            if (i > 50 and cost*(1-eps) < np.mean(costs[-50:]) < cost*(1+eps)) or cost < 1e-6:
                break
            
            if i%10==0:
                costs.append(cost)
                if print_cost:
                    print ("Cost after epoch %i: %f" %(i, cost))
        
        if visualize:
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per 100)')
            plt.title("Learning rate = " + str(self.hyperparam_dict["learning_rate"]))
            plt.show()
        return self
    
    
    def predict(self, X_test):
        n_classes = self.hyperparam_dict["layers_dims"][-1]
        m = X_test.shape[1]
        p = np.zeros((n_classes,m))
        A, _ = self.forward(X_test)
        for i in range(0, A.shape[1]):
            for j in range(0, A.shape[0]):
                if A[j,i] > 0.5:
                    p[j,i] = 1
                else:
                    p[j,i] = 0
        return p
    
    
    def plot_train_test_cost(self, X_train, y_train, X_test, y_test, num_iterations, print_cost=False):
        
        self.__init__(**self.hyperparam_dict)
        self.initialize_parameters(self.hyperparam_dict["layers_dims"])
        eps=1e-5
        train_costs=[]
        test_costs=[]
        seed = self.hyperparam_dict["random_state"]
        best_cost_ = 10e8
        best_model_ = None
        best_iteration_ = 0
        
        if self.hyperparam_dict["optimizer"] == "gd":
            pass
        elif self.hyperparam_dict["optimizer"] == "momentum":
            v = self.initialize_velocity()
        elif self.hyperparam_dict["optimizer"] == "rmsprop":
            s = self.initialize_rmsprop()
        elif self.hyperparam_dict["optimizer"] == "adam":
            v, s = self.initialize_adam()
            t = 0
        
        for i in tqdm_notebook(range(0, num_iterations)):
            seed = seed + 1
            permutation = list(np.random.permutation(X_train.shape[1]))
            train = X_train[:, permutation]
            train_labels = y_train[:, permutation]
            permutation = list(np.random.permutation(X_test.shape[1]))
            test = X_test[:, permutation]
            test_labels = y_test[:, permutation]
            A_train, caches = self.forward(train)
            A_test, _ = self.forward(test)
            cost_train = self.compute_cost(A_train, train_labels)
            cost_test = self.compute_cost(A_test, test_labels)
            grads = self.backward(A_train, train_labels, caches, self.hyperparam_dict["activations"])
                
            if self.hyperparam_dict["optimizer"] == "gd":
                self.update_parameters_with_gd(grads, self.hyperparam_dict["learning_rate"])

            elif self.hyperparam_dict["optimizer"] == "momentum":
                v = self.update_parameters_with_momentum(grads, v, self.hyperparam_dict["beta1"],
                                    self.hyperparam_dict["learning_rate"])

            elif self.hyperparam_dict["optimizer"] == "rmsprop":
                s = self.update_parameters_with_rmsprop(grads, s, self.hyperparam_dict["beta2"],
                                    self.hyperparam_dict["learning_rate"], self.hyperparam_dict["epsilon"])

            elif self.hyperparam_dict["optimizer"] == "adam":
                t += 1
                v, s = self.update_parameters_with_adam(grads, v, s, t,
                                    self.hyperparam_dict["learning_rate"], self.hyperparam_dict["beta1"],
                                    self.hyperparam_dict["beta2"], self.hyperparam_dict["epsilon"])
            
            if cost_test < best_cost_:
                best_cost_ = cost_test
                best_model_ = deepcopy(self)
                best_iteration_ = i
            
            if i%10==0:
                train_costs.append(cost_train)
                test_costs.append(cost_test)
                if print_cost:
                    print ("Cost after epoch %i: on train - %f, \t on test - %f" % (i, cost_train, cost_test))
            
            
        plt.figure(figsize=[8,8])
        plt.plot(np.squeeze(train_costs), label="Train")
        plt.plot(np.squeeze(test_costs), label="Test")
        plt.legend(loc=1)
        plt.ylabel('cost')
        plt.xlabel('iterations (per 100)')
        plt.title("Learning rate = " + str(self.hyperparam_dict["learning_rate"]))
        
        plt.show()
        
        return (best_model_, best_cost_, best_iteration_)
        
        
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()].T)
    preds = np.zeros((1, Z.shape[1]))
    for i in range(0, Z.shape[1]):
        for j in range(0, Z.shape[0]):
            if Z[j, i] == 1:
                preds[0, i] = j
    Z = preds.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)

data_gen, labels_gen = samples_generator(n_samples=5000,
                                 n_features=2,
                                 n_classes=3,
                                 class_sep=True,
                                 ratio=2,
                                 random_state=42)

plt.figure(figsize=[8,8])
plt.scatter(data_gen[:,0], data_gen[:,1], c=labels_gen)
plt.show()