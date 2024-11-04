#include <iostream>

int main() {
    #ifdef __SSE__
        std::cout << "SSE supported" << std::endl;
    #endif
    #ifdef __SSE2__
        std::cout << "SSE2 supported" << std::endl;
    #endif
    #ifdef __SSE4_1__
        std::cout << "SSE4.1 supported" << std::endl;
    #endif
    #ifdef __AVX__
        std::cout << "AVX supported" << std::endl;
    #endif
    #ifdef __AVX2__
        std::cout << "AVX2 supported" << std::endl;
    #endif
    #ifdef __AVX512F__
        std::cout << "AVX-512 supported" << std::endl;
    #endif
    return 0;
}
