#ifndef MAIN_HPP
#define MAIN_HPP

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
#include <complex>
#include <array>
#include <Eigen/Dense>
extern "C" {
#endif

// External interface functions to be used by Python (via pybind11)
size_t get_output_buffer();  // Returns a pointer (cast to size_t) to the raw output buffer.
void set_matrix(size_t dim, size_t pointer);
void set_resolution(size_t resolution);
void set_zoom(double zoom);

// Global variables for the output.
extern size_t width;
extern size_t height;
extern uint32_t *output_buffer;

#ifdef __cplusplus
} // extern "C"

// Internal helper function declarations (for C++ only)
std::array<std::complex<double>, 2> quadratic_solver(double a, double b, double c);
double density_default(Eigen::MatrixXcd &A, Eigen::MatrixXcd &B, double a, double b);
double density_2(Eigen::MatrixXcd &A, Eigen::MatrixXcd &B, double a, double b, double precomp_coeffs[9]);
double cubic_solver_cheat(double values[4]);
double density_3(Eigen::MatrixXcd &A, Eigen::MatrixXcd &B, double a, double b,
                   double precomp_coeffs_1[7][7], double precomp_coeffs_2[7][7],
                   double precomp_coeffs_3[7][7], double precomp_coeffs_4[7][7]);
std::array<std::complex<double>, 3> cubic_solver(double d, double a, double b, double c, size_t &real_roots);
std::array<std::complex<double>, 4> quartic_solver(double e, double a, double b, double c, double d);
double density_4(Eigen::MatrixXcd &A, Eigen::MatrixXcd &B, double a, double b,
                 double precomp_coeffs_1[5][5], double precomp_coeffs_2[5][5],
                 double precomp_coeffs_3[5][5], double precomp_coeffs_4[5][5],
                 double precomp_coeffs_5[5][5]);
void compute_output(Eigen::MatrixXcd &C);
void compute_output_thread(Eigen::MatrixXcd &A, Eigen::MatrixXcd &B, size_t idx);
void compute_output_thread_2(Eigen::MatrixXcd &A, Eigen::MatrixXcd &B, double precomp_coeffs[9], size_t idx);
void compute_output_thread_3(Eigen::MatrixXcd &A, Eigen::MatrixXcd &B,
                             double precomp_coeffs_1[7][7], double precomp_coeffs_2[7][7],
                             double precomp_coeffs_3[7][7], double precomp_coeffs_4[7][7],
                             size_t idx);
void compute_output_thread_4(Eigen::MatrixXcd &A, Eigen::MatrixXcd &B,
                             double precomp_coeffs_1[5][5], double precomp_coeffs_2[5][5],
                             double precomp_coeffs_3[5][5], double precomp_coeffs_4[5][5],
                             double precomp_coeffs_5[5][5],
                             size_t idx);
#endif // __cplusplus

#endif // MAIN_HPP
