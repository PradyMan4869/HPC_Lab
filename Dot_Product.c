#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define VECTOR_LENGTH 1000000000

// Function to generate a vector with random integers from the set {-1, 0, 1}
void generateRandomVector(int *vector, int length) {
    for (int i = 0; i < length; ++i) {
        vector[i] = rand() % 3 - 1; // Random integer from the set {-1, 0, 1}
    }
}

// Function to calculate the dot product of two vectors
int dotProduct(const int *vec1, const int *vec2, int length, int num_threads) {
    int dot = 0;
    #pragma omp parallel for reduction(+:dot) num_threads(num_threads) // Specify number of threads here
    for (int i = 0; i < length; ++i) {
        dot += vec1[i] * vec2[i];
    }
    return dot;
}

// Sequential version of dot product for verification
int sequentialDotProduct(const int *vec1, const int *vec2, int length) {
    int dot = 0;
    for (int i = 0; i < length; ++i) {
        dot += vec1[i] * vec2[i];
    }
    return dot;
}

int main() {
    // Seed the random number generator
    srand(1234);

    // Define vectors
    int *vector1 = (int *)malloc(VECTOR_LENGTH * sizeof(int));
    int *vector2 = (int *)malloc(VECTOR_LENGTH * sizeof(int));

    // Generate random vectors
    generateRandomVector(vector1, VECTOR_LENGTH);
    generateRandomVector(vector2, VECTOR_LENGTH);

    // Specify the number of threads for parallel processing
    int num_threads = 8; // Set the number of threads here

    // Calculate dot product using parallel implementation
    double start_time_parallel = omp_get_wtime();
    int dot_parallel = dotProduct(vector1, vector2, VECTOR_LENGTH, num_threads);
    double end_time_parallel = omp_get_wtime();

    // Calculate dot product using sequential implementation for verification
    double start_time_sequential = omp_get_wtime();
    int dot_sequential = sequentialDotProduct(vector1, vector2, VECTOR_LENGTH);
    double end_time_sequential = omp_get_wtime();

    // Verify if the results are the same
    if (dot_parallel == dot_sequential) {
        printf("Parallel dot product: %d\n", dot_parallel);
        printf("Sequential dot product: %d\n", dot_sequential);
        printf("Results are the same.\n");
    } else {
        printf("Results are different. Parallel: %d, Sequential: %d\n", dot_parallel, dot_sequential);
    }

    // Print execution times
    printf("Parallel execution time: %f seconds\n", end_time_parallel - start_time_parallel);
    printf("Sequential execution time: %f seconds\n", end_time_sequential - start_time_sequential);

    // Free allocated memory
    free(vector1);
    free(vector2);

    return 0;
}
