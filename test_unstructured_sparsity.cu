#include "include/utils.cuh"
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <ctime>

using std::cout;
using std::endl;

// Helper to count actual zeros in a matrix
int count_zeros(float* mat, int rows, int cols) {
    int zeros = 0;
    for (int i = 0; i < rows * cols; i++) {
        if (mat[i] == 0.0f) zeros++;
    }
    return zeros;
}

// Helper to print small matrix (for visualization)
void print_matrix_snippet(float* mat, int rows, int cols, int max_rows = 8, int max_cols = 8) {
    cout << "Matrix snippet (" << rows << "×" << cols << "):\n";
    for (int i = 0; i < std::min(rows, max_rows); i++) {
        for (int j = 0; j < std::min(cols, max_cols); j++) {
            if (mat[i * cols + j] == 0.0f) {
                cout << "  ·  ";  // Zero element
            } else {
                cout << std::setw(5) << std::fixed << std::setprecision(1)
                     << mat[i * cols + j] << " ";
            }
        }
        if (cols > max_cols) cout << " ...";
        cout << "\n";
    }
    if (rows > max_rows) cout << " ...\n";
    cout << endl;
}

int main(int argc, char** argv) {
    cout << "========================================\n";
    cout << "Unstructured Random Sparsity Test\n";
    cout << "========================================\n\n";

    // Parse command line arguments
    int size = 1024;
    float sparsity = 50.0f;
    unsigned int seed = 12345;

    if (argc > 1) size = atoi(argv[1]);
    if (argc > 2) sparsity = atof(argv[2]);
    if (argc > 3) seed = atoi(argv[3]);

    cout << "Configuration:\n";
    cout << "  Matrix size: " << size << "×" << size << "\n";
    cout << "  Target sparsity: " << sparsity << "%\n";
    cout << "  Random seed: " << seed << "\n\n";

    // Test 1: Single matrix with unstructured sparsity
    cout << "=== Test 1: Single Matrix Unstructured Sparsity ===\n";
    {
        float* mat = new float[size * size];
        randomize_matrix_unstructured(mat, size, size, sparsity, seed);

        int zeros = count_zeros(mat, size, size);
        float actual_sparsity = 100.0f * zeros / (size * size);

        cout << "Actual sparsity: " << std::fixed << std::setprecision(2)
             << actual_sparsity << "% (" << zeros << "/" << (size*size) << " zeros)\n";

        if (size <= 16) {
            print_matrix_snippet(mat, size, size, size, size);
        } else {
            print_matrix_snippet(mat, size, size, 8, 8);
        }

        delete[] mat;
    }

    // Test 2: Joint A+B unstructured sparsity
    cout << "=== Test 2: Joint A+B Unstructured Sparsity ===\n";
    {
        float sparsity_A = 50.0f;
        float sparsity_B = 50.0f;

        if (argc > 4) sparsity_A = atof(argv[4]);
        if (argc > 5) sparsity_B = atof(argv[5]);

        int M = size;
        int K = size;
        int N = size;

        float* mat_A = new float[M * K];
        float* mat_B = new float[K * N];

        randomize_matrices_joint_unstructured(mat_A, mat_B, M, N, K,
                                              sparsity_A, sparsity_B, seed);

        int zeros_A = count_zeros(mat_A, M, K);
        int zeros_B = count_zeros(mat_B, K, N);

        float actual_A = 100.0f * zeros_A / (M * K);
        float actual_B = 100.0f * zeros_B / (K * N);

        cout << "\nActual sparsity A: " << std::fixed << std::setprecision(2)
             << actual_A << "% (" << zeros_A << "/" << (M*K) << " zeros)\n";
        cout << "Actual sparsity B: " << actual_B << "% ("
             << zeros_B << "/" << (K*N) << " zeros)\n";

        // Calculate theoretical joint sparsity
        float density_A = (100.0f - actual_A) / 100.0f;
        float density_B = (100.0f - actual_B) / 100.0f;
        float joint_density = density_A * density_B;

        cout << "\nTheoretical joint density: " << (joint_density * 100.0f)
             << "% (expect " << (100.0f * (1.0f - joint_density))
             << "% effective sparsity)\n";

        if (size <= 16) {
            cout << "\nMatrix A:\n";
            print_matrix_snippet(mat_A, M, K, M, K);
            cout << "Matrix B:\n";
            print_matrix_snippet(mat_B, K, N, K, N);
        }

        delete[] mat_A;
        delete[] mat_B;
    }

    // Test 3: Different sparsity levels
    cout << "\n=== Test 3: Sparsity Level Sweep ===\n";
    cout << std::setw(15) << "Target %"
         << std::setw(15) << "Actual %"
         << std::setw(15) << "Error %"
         << std::setw(15) << "Zeros\n";
    cout << std::string(60, '-') << "\n";

    int test_size = 256;
    float sparsity_levels[] = {10.0f, 25.0f, 50.0f, 75.0f, 90.0f, 95.0f, 99.0f};

    for (float test_sparsity : sparsity_levels) {
        float* mat = new float[test_size * test_size];
        randomize_matrix_unstructured(mat, test_size, test_size, test_sparsity, seed);

        int zeros = count_zeros(mat, test_size, test_size);
        float actual = 100.0f * zeros / (test_size * test_size);
        float error = actual - test_sparsity;

        cout << std::setw(15) << std::fixed << std::setprecision(1) << test_sparsity
             << std::setw(15) << actual
             << std::setw(15) << error
             << std::setw(15) << zeros << "\n";

        delete[] mat;
    }

    // Test 4: Reproducibility test
    cout << "\n=== Test 4: Reproducibility Test ===\n";
    {
        int test_size = 100;
        float* mat1 = new float[test_size * test_size];
        float* mat2 = new float[test_size * test_size];

        // Generate with same seed
        randomize_matrix_unstructured(mat1, test_size, test_size, 50.0f, 42);
        randomize_matrix_unstructured(mat2, test_size, test_size, 50.0f, 42);

        bool identical = true;
        for (int i = 0; i < test_size * test_size; i++) {
            if (mat1[i] != mat2[i]) {
                identical = false;
                break;
            }
        }

        cout << "Same seed (42): " << (identical ? "✓ IDENTICAL" : "✗ DIFFERENT") << "\n";

        // Generate with different seed
        randomize_matrix_unstructured(mat2, test_size, test_size, 50.0f, 99);

        identical = true;
        for (int i = 0; i < test_size * test_size; i++) {
            if (mat1[i] != mat2[i]) {
                identical = false;
                break;
            }
        }

        cout << "Different seed (42 vs 99): " << (identical ? "✗ IDENTICAL (bad)" : "✓ DIFFERENT") << "\n";

        delete[] mat1;
        delete[] mat2;
    }

    cout << "\n========================================\n";
    cout << "All tests completed successfully!\n";
    cout << "========================================\n";

    return 0;
}
