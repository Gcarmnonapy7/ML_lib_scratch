#ifndef KNN_HPP
#define KNN_HPP
#include <cmath>
#include <vector>
#include <stdexcept>
#include <algorithm>

class KNN {
    private:
        int k;
        std::vector<std::vector<double>> X_train;
        std::vector<int> Y_train;

        double euclideanDistance(const std::vector<double>&a , const std::vector<double>&b){
            if (a.size() != b.size()){
                throw std::invalid_argument("Vectors must be of the same dimension");}
            double sum = 0.0;
            for (size_t i = 0; i < a.size(); i++){
                sum += (a[i] - b[i]) * (a[i] - b[i]);
            }
            return std::sqrt(sum); // Implementation of Euclidean distance formula
        }

    public:
        int k = 3; // Default value
        
        KNN(int k) : k(k) {
            if (k <= 0){
                throw std::invalid_argument("k must be a positive integer"); 
            } 
        }
        void fit (const std::vector<std::vector<double>>& X, const std::vector<int>& Y){
            if (X.size() != Y.size()){
                throw std::invalid_argument("The number of samples in X must match the number of Labels");
            } // Check if the sizes of X and Y match
            else if (X.empty()) {
                throw std::invalid_argument("Training data X cannot be empty");
            }
            //Store the traning data 
            this->X_train = X;
            this->Y_train = Y;
        }
        void predict (const std::vector<std::vector<double>>& X, std::vector<int>& predictions){
            if (X.empty()){
                throw std::invalid_argument("X input cannot be empty");
            }
            if (this->X_train.empty() || this->Y_train.empty()){
                throw std::logic_error("Model has not been fitted with training data");
            }
            predictions.clear(); // Ensure predictions vector is empty

            for (const auto& x :X){
                std::vector<std::pair<double,int>> distances;
                for (size_t i=0; i < this->X_train.size(); i++)
                {
                    // Calculate distance between x and each point in X_train
                    double distance = euclideanDistance(x, this->X_train[i]);
                    distances.push_back({distance, this->Y_train[i]}); // Store distance and corresponding label using the push_back method
                }; 
                // Sort distances to find the k nearest neighbors
                std::sort(distances.begin(), distances.end());
            }
        }
};
#endif // KNN_HPP