//
// Created by Jozef Karabelly (xkarab03)
//
#ifndef SFC_HELPERS_H
#define SFC_HELPERS_H

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>


namespace helpers {
    /**
    * Pretty print an C-like array of values
    * @param  data   Array of values
    * @param  length Size of the array
    * @return        String representation of the array
    */
    template<class T>
    inline std::string printVector(T *data, size_t length) {
        std::string output = "[";
        for (unsigned x = 0; x < length; x++) {
            output += std::to_string(data[x]) + ", ";
        }
        output.erase(output.end() - 2, output.end());
        output += "]";
        return output;
    }

    /**
     * Print heading with separator
     * @param heading Heading text
     */
    inline void header(std::string heading) {
        std::cout << std::endl << heading << std::endl << std::string(80, '-') << std::endl;
    }

    /**
     * Handle waiting for user input and user response
     * @return True if user wants to continue else False
     */
    inline bool waitToContinue() {
        bool ret = true;
        std::cout << "Skip to the end (s) | Else continue (Enter)" << std::endl;
        int flag = std::cin.get();
        if (flag == 's') {
            ret = false;
        }
        return ret;
    }
}

#endif //SFC_HELPERS_H
