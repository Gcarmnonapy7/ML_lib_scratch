#include <cmath>
#include <vector>
#include <stdexcept>

double euclideanDistance(double x1, double y1,double x2,double y2) {
    if (std::isnan(x1) || std::isnan(y1) || std::isnan(x2) || std::isnan(y2)) {
        throw std::invalid_argument("Input coordinates must not be NaN");
    }
    double dx = x2 - x1;
    double dy = y2 - y1;
    return std::sqrt((dx * dx)+(dy * dy));
}

class KNN {
    private:
        int k;
    public:
        KNN(int k) : k(k) {
            if (k <= 0){
                throw std::invalid_argument("k must be a positive integer"); 
            } 
        }
        void fit(){}
};