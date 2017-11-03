from terms import *


class MyModelLR:
    def __init__(self):
        self.weights = None
        self.inputs = None
        self.targets = None

    def predict(self, x):
        return sigmoid(u(x, self.inputs, self.weights, self.c))

    def train(self, X, y, c, lm, eta=10, it=20):
        self.inputs = X
        self.targets = y
        self.c = c
        self.lm = lm
        self.eta = eta

        print 'Training model...'
        n = len(X)
        w = np.zeros(n)
        iterations = 0
        while iterations < it:
            # Build terms
            J = np.zeros((n, n))
            fB = np.zeros(y.shape)
            for i, x in enumerate(X):
                J[i] = grad_g(x, X, w, c, lm)
                fB[i] = g(x, X, w, c, lm)

            # Solve for delta
            p = np.matmul(J.T, J)
            lhs = p + eta * np.diag(np.diag(p))
            rhs = np.dot(J.T, y - fB)
            inv = np.linalg.inv(lhs)
            delta = np.dot(inv, rhs)
            w_old = w
            w = w + delta
            iterations += 1
            print iterations
            print np.linalg.norm(w_old - w)
        self.weights = w
