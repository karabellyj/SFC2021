//
// Created by Jozef Karabelly (xkarab03)
//
#include <string>
#include <iostream>
#include <ctime>
#include <algorithm>
#include "rbm.h"

// default settings for the network
#define LEARNING_RATE 0.1
#define VISIBLE_NEURONS 6
#define HIDDEN_NEURONS 3
#define EPOCHS 1

/*
 * Input parameter parser
 */
class InputParser {
public:
    InputParser(int &argc, char **argv) {
        for (int i = 1; i < argc; ++i)
            this->tokens.push_back(std::string(argv[i]));
    }

    /**
     *
     * @param option name of the program parameter
     * @return if exists input value else empty string
     */
    std::string getCmdOption(const std::string &option) const {
        std::vector<std::string>::const_iterator itr;
        itr = std::find(this->tokens.begin(), this->tokens.end(), option);
        if (itr != this->tokens.end() && ++itr != this->tokens.end()) {
            return *itr;
        }
        static const std::string empty_string("");
        return empty_string;
    }

private:
    std::vector<std::string> tokens;
};

int main(int argc, char **argv) {
    // initialize random module
    srand(time(NULL));

    // parse inputs
    InputParser input(argc, argv);

    // set default values for algorithm parameters
    int numVisible = VISIBLE_NEURONS, numHidden = HIDDEN_NEURONS;
    int epochs = EPOCHS;
    double learningRate = LEARNING_RATE;

    // check if any values were pass using arguments
    std::string hiddenValue = input.getCmdOption("--numHidden");
    std::string lrValue = input.getCmdOption("--learningRate");
    std::string epochValue = input.getCmdOption("--epochs");
    if (!hiddenValue.empty())
        numHidden = std::stoi(hiddenValue);
    if (!lrValue.empty())
        learningRate = std::stod(lrValue);
    if (!epochValue.empty())
        epochs = std::stoi(epochValue);

    // construct RBM
    RBM rbm(numVisible, numHidden, learningRate, true);
    rbm.printState();

    // initializing training data
    helpers::header("Initializing training data");
    std::vector<std::vector<int>> trainingData{
            {1, 1, 1, 0, 0, 0},
            {1, 0, 1, 0, 0, 0},
            {1, 1, 1, 0, 0, 0},
            {0, 0, 0, 1, 1, 1},
            {0, 0, 0, 1, 0, 1},
            {0, 0, 0, 1, 1, 1}
    };
    for (auto x: trainingData) std::cout << helpers::printVector(x.data(), numVisible) << std::endl;

    // training of the network
    helpers::header("Trained network (" + std::to_string(epochs) + " epochs)");
    rbm.train(trainingData, epochs);
    rbm.printState();

    return 0;
}
