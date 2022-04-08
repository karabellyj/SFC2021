//
// Created by Jozef Karabelly (xkarab03)
//
#include <math.h>
#include <random>
#include <iostream>
#include <iomanip>
#include <vector>
#include "helpers.h"

#ifndef SFC_RBM_H
#define SFC_RBM_H


class RBM {
public:
    /**
     * Initializes model weights
     * @param numVisible number of visible neurons
     * @param numHidden number of hidden neurons
     * @param learningRate learning rate parameter for weight update
     * @param interactive prints states if True else runs to the end
     */
    RBM(unsigned numVisible, unsigned numHidden, double learningRate, bool interactive);

    ~RBM();

    /**
     * Prints weights in pretty format
     */
    void printState();

    /**
     * Trains RBM using contrastive divergence
     * @param input training dataset
     * @param numEpochs number of epochs
     */
    void train(std::vector<std::vector<int>> input, unsigned numEpochs);

protected:
    /**
     * Sigmoid function
     * @param a sigmoid parameter
     * @return sigmoid activation
     */
    inline float sigmoid(double a) {
        return 1.0 / (1.0 + exp(-a));
    }

    /**
    * Compute activation based on probability
    * @param  sample probability
    * @return        binary activation
    */
    inline int binomial(double sample) {
        if (sample < 0 || sample > 1) return 0;
        if (sample > rand() / (RAND_MAX + 1.0)) return 1;
        return 0;
    }

private:
    unsigned numVisible;
    unsigned numHidden;
    double learningRate;
    double **weights;
    bool interactive;

    /**
     * Calculates probability p(a|b)
     * @param b vector of hidden layer values
     * @param a vector of visible layer values
     */
    void probabilityOfA(int *, int *);
    /**
     * Calculates probability p(b|a)
     * @param a vector of visible layer values
     * @param b vector of hidden layer values
     */
    void probabilityOfB(int *, int *);

    /**
     * Calculates probability using sigmoid activation
     * @param a vector of visible layer values
     * @param weights vector of weights for current b
     * @return sigmoid activation for p(b|a)
     */
    double propagateFromVisible(int *, double *);

    /**
     * Calculates probability using sigmoid activation
     * @param b vector of hidden layer values
     * @param i index of current a
     * @return sigmoid activation for p(a|b)
     */
    double propagateFromHidden(int *, unsigned);

    /**
     * Initializes probabilities to 0
     * @return  matrix of probabilities
     */
    std::vector<std::vector<double>> initializeProbabilities() const;

    /**
     * Helper method to return bias of visible neurons from weights matrix
     * @return vector of biases for visible neurons
     */
    std::vector<double> getVisibleBias();

    /**
     * Helper method to return bias of hidden neurons form weights matrix
     * @return vecotr of biases for hidden neurons
     */
    std::vector<double> getHiddenBias();

    /**
     * Updates probability matrix
     * @param proba probability matrix
     * @param a vector of visible neuron values
     * @param b vector of hidden neuron values
     * @param P number of training samples
     */
    void updateProbabilities(std::vector<std::vector<double>> &proba, int *a, int *b, int P);
};


#endif //SFC_RBM_H
