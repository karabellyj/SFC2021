//
// Created by Jozef Karabelly (xkarab03)
//
#include "rbm.h"

RBM::RBM(unsigned int numVisible, unsigned int numHidden, double learningRate, bool interactive) :
        numVisible(numVisible),
        numHidden(numHidden),
        learningRate(learningRate),
        interactive(interactive) {
    // initialize normal distribution generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> normal(0, 1);

    // initialize weights with numbers from normal distribution
    weights = new double *[numVisible + 1];
    for (unsigned h = 0; h < numHidden + 1; ++h) {
        weights[h] = new double[numVisible + 1];
        for (unsigned v = 0; v < numVisible + 1; ++v) {
            weights[h][v] = normal(gen);
        }
    }
}

RBM::~RBM() {
    for (unsigned i = 0; i < numHidden + 1; ++i) {
        delete[] weights[i];
    }
    delete[] weights;
}

void RBM::probabilityOfB(int *a, int *activations) {
    activations[0] = 1; // bias
    for (unsigned j = 1; j < numHidden + 1; ++j) {
        activations[j] = binomial(propagateFromVisible(a, weights[j]));
    }
}

void RBM::probabilityOfA(int *b, int *activations) {
    activations[0] = 1; // bias
    for (unsigned i = 1; i < numVisible + 1; ++i) {
        activations[i] = binomial(propagateFromHidden(b, i));
    }
}

double RBM::propagateFromVisible(int *a, double *w) {
    double temp = 0;
    for (unsigned i = 0; i < numVisible + 1; ++i) {
        temp += w[i] * a[i];
    }
    return sigmoid(temp);
}

double RBM::propagateFromHidden(int *b, unsigned i) {
    double temp = 0;
    for (unsigned j = 0; j < numHidden + 1; ++j) {
        temp += weights[j][i] * b[j];
    }
    return sigmoid(temp);
}

std::vector<std::vector<double>> RBM::initializeProbabilities() const {
    std::vector<std::vector<double>> matrix(numHidden + 1);
    for (unsigned i = 0; i < numHidden + 1; i++)
        matrix[i].resize(numVisible + 1);
    return matrix;
}

void RBM::updateProbabilities(std::vector<std::vector<double>> &proba, int *a, int *b, int P) {
    for (unsigned h = 0; h < numHidden + 1; ++h) {
        for (unsigned v = 0; v < numVisible + 1; ++v) {
            if (b[h] == a[v])
                proba[h][v] += 1.0 / P;
        }
    }
}

void RBM::train(std::vector<std::vector<int>> input, unsigned int numEpochs) {
    std::vector<int> negative_activation_hidden(numHidden + 1);
    std::vector<int> positive_activation_hidden(numHidden + 1);
    std::vector<int> negative_activation_visible(numVisible + 1);

    for (unsigned e = 0; e < numEpochs; ++e) {
        std::vector<std::vector<double>> probas1 = initializeProbabilities();
        std::vector<std::vector<double>> probas2 = initializeProbabilities();
        for (unsigned i = 0; i < input.size(); ++i) {
            // a positive phase of the contrastive divergence
            if (interactive)
                helpers::header("Positive phase of contrastive divergence");
            // add 1 as the first element of input and set a = input
            std::vector<int> a(1);
            a[0] = 1;
            a.insert(a.end(), input[i].begin(), input[i].end());

            if (interactive) {
                std::cout << "a = ";
                std::cout << helpers::printVector(a.data(), a.size()) << std::endl;
                interactive = helpers::waitToContinue();
            }

            if (interactive)
                std::cout << "Calculating probability p(b|a)" << std::endl;

            // calculate positive hidden activation probabilities
            probabilityOfB(a.data(), positive_activation_hidden.data());

            if (interactive) {
                std::cout << "b = ";
                std::cout << helpers::printVector(positive_activation_hidden.data(), positive_activation_hidden.size())
                          << std::endl;
                interactive = helpers::waitToContinue();
                std::cout << "Updating probabilities 1p" << std::endl;
            }

            // update probabilities
            updateProbabilities(probas1, a.data(), positive_activation_hidden.data(), input.size());

            if (interactive) {
                std::cout << "1p = " << std::endl;
                for (auto row: probas1) {
                    std::cout << helpers::printVector(row.data(), row.size()) << std::endl;
                }
                interactive = helpers::waitToContinue();
            }

            // a negative (reconstruction) phase of the contrastive divergence
            if (interactive) {
                helpers::header("Negative phase of the contrastive divergence");
                std::cout << "Calculating probability p(a|b)" << std::endl;
            }

            // calculate negative visible probabilities
            probabilityOfA(positive_activation_hidden.data(), negative_activation_visible.data());

            if (interactive) {
                std::cout << "a = ";
                std::cout
                        << helpers::printVector(negative_activation_visible.data(), negative_activation_visible.size())
                        << std::endl;
                interactive = helpers::waitToContinue();
                std::cout << "Calculating probability p(b|a)" << std::endl;
            }

            // negative hidden probabilities
            probabilityOfB(negative_activation_visible.data(), negative_activation_hidden.data());

            if (interactive) {
                std::cout << "b = ";
                std::cout << helpers::printVector(negative_activation_hidden.data(), negative_activation_hidden.size())
                          << std::endl;
                interactive = helpers::waitToContinue();
                std::cout << "Updating probabilities 2p" << std::endl;
            }

            // update probabilities
            updateProbabilities(probas2, negative_activation_visible.data(), negative_activation_hidden.data(),
                                input.size());

            if (interactive) {
                std::cout << "2p = " << std::endl;
                for (auto row: probas2) {
                    std::cout << helpers::printVector(row.data(), row.size()) << std::endl;
                }
                interactive = helpers::waitToContinue();
                std::cout << "Updating weights" << std::endl;
            }

            // update weights
            for (unsigned h = 0; h < numHidden + 1; ++h) {
                for (unsigned v = 0; v < numVisible + 1; ++v) {
                    weights[h][v] += learningRate * (probas1[h][v] - probas2[h][v]);
                }
            }

            if (interactive) {
                std::cout << "w = " << std::endl;
                for (unsigned i = 0; i < numHidden + 1; ++i) {
                    std::cout << helpers::printVector(weights[i], numVisible + 1) << std::endl;
                }
                interactive = helpers::waitToContinue();
            }
        }
    }
}

void RBM::printState() {
    std::cout << std::fixed << std::setprecision(4);

    std::cout << "Bias of visible nodes (" << numVisible << "):\t";
    std::cout << helpers::printVector(getVisibleBias().data(), numVisible) << std::endl;
    std::cout << "Bias of hidden nodes (" << numHidden << "):\t";
    std::cout << helpers::printVector(getHiddenBias().data(), numHidden) << std::endl;


    std::cout << "Weights:" << std::endl << std::setw(12) << std::setfill(' ') << "|";
    for (unsigned v = 1; v < numVisible + 1; v++) std::cout << std::setw(7) << std::setfill(' ') << "#" << v;
    std::cout << std::endl << std::string(12 + numVisible * 8, '-') << std::endl;
    for (unsigned h = 1; h < numHidden + 1; h++) {
        std::cout << std::setw(9) << std::setfill(' ') << "Hidden #" << h << " |";
        for (unsigned v = 1; v < numVisible + 1; v++) std::cout << std::setw(8) << std::setfill(' ') << weights[h][v];
        std::cout << std::endl;
    }
}

std::vector<double> RBM::getVisibleBias() {
    std::vector<double> visibleBias(numVisible);
    for (unsigned i = 1; i < numVisible + 1; ++i) {
        visibleBias[i - 1] = weights[0][i];
    }
    return visibleBias;
}

std::vector<double> RBM::getHiddenBias() {
    std::vector<double> hiddenBias(numHidden);
    for (unsigned j = 1; j < numHidden + 1; ++j) {
        hiddenBias[j - 1] = weights[j][0];
    }
    return hiddenBias;
}



