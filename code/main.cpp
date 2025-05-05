// main.cpp
#include <cmath>
#include <complex>
#include <cstdlib>
#include <Eigen/Dense>
#include <iostream>
#include <thread>
#include <vector>
#include <array>
#include <cstdint>
#include <cstring>
#include "main.hpp"

// Global variables.
size_t width = 0, height = 0;
size_t num_threads = std::thread::hardware_concurrency();
double *output = nullptr;
uint32_t *output_buffer = nullptr;
int d;
double p_scale = 3;
double zoom_scale = 1;
double dr_scale = 1;

const std::complex<double> If(0.0, 1.0);
const std::complex<double> Omega(-1.0/2.0, std::sqrt(3.0) / 2);
const std::complex<double> Omega_sq(-1.0/2.0, -std::sqrt(3.0) / 2);
const double eps = 1e-12;
const double PI = 3.141592653589793238463;
const double M_2PI = 2 * PI;


#ifdef BUILD_PYTHON
extern "C" {
#endif

size_t get_output_buffer() {
    return reinterpret_cast<size_t>(output_buffer);
}

#ifdef BUILD_PYTHON
}
#endif

//------------------------------------------------------------------------------
// Forward declarations.
std::array<std::complex<double>, 2> quadratic_solver(double a, double b, double c);
double density_default(Eigen::MatrixXcd &A, Eigen::MatrixXcd &B, double a, double b);
double density_2(Eigen::MatrixXcd &A, Eigen::MatrixXcd &B, double a, double b, double precomp_coeffs[9]);
double cubic_solver_cheat(double values[4]);
double density_3(Eigen::MatrixXcd &A, Eigen::MatrixXcd &B, double a, double b,
                   double precomp_coeffs_1[7][7], double precomp_coeffs_2[7][7],
                   double precomp_coeffs_3[7][7], double precomp_coeffs_4[7][7]);
std::array<std::complex<double>, 3> cubic_solver(double d, double a, double b, double c, size_t &real_roots);
std::array<std::complex<double>, 4> quartic_solver(double e, double a, double b, double c, double d);
// Forward declaration for density_4.
double density_4(Eigen::MatrixXcd &A, Eigen::MatrixXcd &B, double a, double b,
                 double precomp_coeffs_1[5][5], double precomp_coeffs_2[5][5],
                 double precomp_coeffs_3[5][5], double precomp_coeffs_4[5][5],
                 double precomp_coeffs_5[5][5]);

//------------------------------------------------------------------------------
// quadratic_solver: Solves ax^2 + bx + c = 0.
std::array<std::complex<double>, 2> quadratic_solver(double a, double b, double c) {
    if (std::abs(a) < eps) {
        return {0, 0};
    }
    auto D = b * b - 4 * a * c;
    if (D > eps) {
        return { (-b + std::sqrt(D)) / (2 * a),
                 (-b - std::sqrt(D)) / (2 * a) };
    } else if (D < -eps) {
        return { (-b + If * std::sqrt(-D)) / (2 * a),
                 (-b - If * std::sqrt(-D)) / (2 * a) };
    } else {
        return { (-b) / (2 * a), (-b) / (2 * a) };
    }
}

//------------------------------------------------------------------------------
// density_default: Computes density via an eigenvalue approach.
double density_default(Eigen::MatrixXcd &A, Eigen::MatrixXcd &B, double a, double b) {
    Eigen::MatrixXcd first = Eigen::MatrixXcd::Identity(A.rows(), A.cols()) -
        (((a / (a * a + b * b)) * A) + (b / (a * a + b * b)) * B);
    Eigen::MatrixXcd second = (b * A - a * B).inverse();
    Eigen::ComplexEigenSolver<Eigen::MatrixXcd> eigensolver(first * second, false);
    if (eigensolver.info() != Eigen::Success) { 
        return 0; 
    }
    auto eigenvalues = eigensolver.eigenvalues();
    double value = 0;
    for (int i = 0; i < eigenvalues.size(); i++) {
        if (std::abs(eigenvalues[i].imag()) > eps) {
            value += std::abs(eigenvalues[i].imag());
        }
    }
    return value;
}

//------------------------------------------------------------------------------
// density_2: Computes density using precomputed coefficients.
double density_2(Eigen::MatrixXcd &A, Eigen::MatrixXcd &B, double a, double b, double precomp_coeffs[9]) {
    double value = precomp_coeffs[0] + precomp_coeffs[1]*b + precomp_coeffs[2]*b*b +
                   precomp_coeffs[3]*a + precomp_coeffs[4]*a*b + precomp_coeffs[5]*a*a;
    if (value < eps) { return 0.0; }
    double x = (precomp_coeffs[6]*b*b + precomp_coeffs[7]*a*b + precomp_coeffs[8]*a*a);
    value /= (x * x);
    return std::sqrt(value);
}

//------------------------------------------------------------------------------
// cubic_solver_cheat: Helper for solving a reduced cubic equation.
double cubic_solver_cheat(double values[4]) {
    if (std::abs(values[3]) < eps) { 
        if (std::abs(values[2]) < eps) {
            return 0;
        } 
        auto D = values[1] * values[1] - 4 * values[0] * values[2];
        if (D > eps) { return 0; }
        return std::sqrt(-D) / (2.0 * values[2]);
    }
    auto p = -values[2] / (3.0 * values[3]);
    auto q = p*p*p + (values[1]*values[2] - 3.0*values[0]*values[3]) / (6.0*(values[3]*values[3]));
    auto r = values[1] / (3.0*values[3]);
    auto rp2 = (r - p*p);
    auto h = q*q + rp2*rp2*rp2;
    if (h < eps) { return 0; }
    h = std::sqrt(h);
    return std::sqrt(3.0) * std::abs(std::cbrt(q + h) - std::cbrt(q - h)) / 2.0;
}

//------------------------------------------------------------------------------
// density_3: Computes density using four sets of precomputed 7x7 coefficient arrays.
double density_3(Eigen::MatrixXcd &A, Eigen::MatrixXcd &B, double a, double b,
                 double precomp_coeffs_1[7][7], double precomp_coeffs_2[7][7],
                 double precomp_coeffs_3[7][7], double precomp_coeffs_4[7][7]) {
    double values[4] = {0, 0, 0, 0};
    double a_powers[7] = {1, a, 0, 0, 0, 0, 0};
    double b_powers[7] = {1, b, 0, 0, 0, 0, 0};
    for (size_t i = 2; i < 7; i++) {
        a_powers[i] = a_powers[i - 1] * a;
        b_powers[i] = b_powers[i - 1] * b;
    }
    for (size_t i = 0; i < 7; i++) {
        for (size_t j = 0; j < 7; j++) {
            double prod = a_powers[j] * b_powers[i];
            values[0] += precomp_coeffs_1[i][j] * prod;
            values[1] += precomp_coeffs_2[i][j] * prod;
            values[2] += precomp_coeffs_3[i][j] * prod;
            values[3] += precomp_coeffs_4[i][j] * prod;
        }
    }
    return cubic_solver_cheat(values) / (a*a + b*b);
}

//------------------------------------------------------------------------------
// cubic_solver: Solves a cubic equation d*x^3 + a*x^2 + b*x + c = 0.
std::array<std::complex<double>, 3> cubic_solver(double d, double a, double b, double c, size_t &real_roots) {
    if (std::abs(d) < eps) { 
        auto roots = quadratic_solver(a, b, c);
        real_roots = 2;
        return { roots[0], roots[1], 0 };
    }
    a /= d; b /= d; c /= d;
    auto a2 = a * a;
    auto q = (a2 - 3*b) / 9;
    auto r = (a*(2*a2 - 9*b) + 27*c) / 54;
    auto r2 = r * r;
    auto q3 = q * q * q;
    if (r2 < q3) {
        auto t = r / std::sqrt(q3);
        if (t < -1) { t = -1; }
        else if (t > 1) { t = 1; }
        t = std::acos(t);
        a /= 3;
        q = -2 * std::sqrt(q);
        real_roots = 3;
        return { q * std::cos(t/3) - a,
                 q * std::cos((t + M_2PI)/3) - a,
                 q * std::cos((t - M_2PI)/3) - a };
    } else {
        double A = -std::pow(std::abs(r) + std::sqrt(r2 - q3), 1.0/3);
        double B;
        if (r < 0) { A = -A; }
        if (std::abs(A) < eps) { B = 0; } else { B = q / A; }
        a /= 3;
        double imaginary_part = 0.5 * std::sqrt(3.0) * (A - B);
        double real_part = -0.5 * (A + B) - a;
        if (std::abs(imaginary_part) < eps) {
            real_roots = 3;
            return { (A + B) - a, real_part, real_part };
        } else {
            real_roots = 1;
            return { (A + B) - a,
                     real_part + If * imaginary_part,
                     real_part - If * imaginary_part };
        }
    }
}

//------------------------------------------------------------------------------
// quartic_solver: Solves a quartic equation e*x^4 + a*x^3 + b*x^2 + c*x + d = 0.
std::array<std::complex<double>, 4> quartic_solver(double e, double a, double b, double c, double d) {
    if (std::abs(e) < eps) {
        size_t dummy;
        auto roots = cubic_solver(a, b, c, d, dummy);
        return { roots[0], roots[1], roots[2], 0 };
    }
    a /= e; b /= e; c /= e; d /= e;
    double a3 = -b;
    double b3 = a * c - 4.0 * d;
    double c3 = -a * a * d - c * c + 4.0 * b * d;
    size_t iZeroes = 0;
    auto x3 = cubic_solver(1, a3, b3, c3, iZeroes);
    double q1, q2, p1, p2, D, sqD, y = x3[0].real();
    if (iZeroes > 1) {
        if (std::abs(x3[1].real()) > std::abs(y)) y = x3[1].real();
        if (std::abs(x3[2].real()) > std::abs(y)) y = x3[2].real();
    }
    D = y * y - 4 * d;
    if (std::abs(D) < eps) {
        q1 = q2 = y * 0.5;
        D = a * a - 4 * (b - y);
        if (std::abs(D) < eps) {
            p1 = p2 = a * 0.5;
        } else {
            sqD = std::sqrt(D);
            p1 = (a + sqD) * 0.5;
            p2 = (a - sqD) * 0.5;
        }
    } else {
        sqD = std::sqrt(D);
        q1 = (y + sqD) * 0.5;
        q2 = (y - sqD) * 0.5;
        p1 = (a * q1 - c) / (q1 - q2);
        p2 = (c - a * q2) / (q1 - q2);
    }
    auto first = quadratic_solver(1, p1, q1);
    auto second = quadratic_solver(1, p2, q2);
    return { first[0], first[1], second[0], second[1] };
}

//------------------------------------------------------------------------------
// Thread worker functions.

// compute_output_thread: Uses density_default.
void compute_output_thread(Eigen::MatrixXcd &A, Eigen::MatrixXcd &B, size_t idx) {
    size_t start = (height / num_threads) * idx;
    size_t end = (height / num_threads) * (idx + 1);
    for (size_t i = start; i < end; i++) {
        for (size_t j = 0; j < width; j++) {
            output[i * width + j] = density_default(A, B,
                dr_scale * ((j * 2.0 / width) - 1.0),
                -dr_scale * ((i * 2.0 / height) - 1.0));
        }
    }
}

// compute_output_thread_2: Uses density_2.
void compute_output_thread_2(Eigen::MatrixXcd &A, Eigen::MatrixXcd &B, double precomp_coeffs[9], size_t idx) {
    size_t start = (height / num_threads) * idx;
    size_t end = (height / num_threads) * (idx + 1);
    for (size_t i = start; i < end; i++) {
        for (size_t j = 0; j < width; j++) {
            output[i * width + j] = density_2(A, B,
                dr_scale * ((j * 2.0 / width) - 1.0),
                -dr_scale * ((i * 2.0 / height) - 1.0), precomp_coeffs);
        }
    }
}

// compute_output_thread_3: Uses density_3.
void compute_output_thread_3(Eigen::MatrixXcd &A, Eigen::MatrixXcd &B,
                             double precomp_coeffs_1[7][7], double precomp_coeffs_2[7][7],
                             double precomp_coeffs_3[7][7], double precomp_coeffs_4[7][7],
                             size_t idx) {
    size_t start = (height / num_threads) * idx;
    size_t end = (height / num_threads) * (idx + 1);
    for (size_t i = start; i < end; i++) {
        for (size_t j = 0; j < width; j++) {
            output[i * width + j] = density_3(A, B,
                dr_scale * ((j * 2.0 / width) - 1.0),
                -dr_scale * ((i * 2.0 / height) - 1.0),
                precomp_coeffs_1, precomp_coeffs_2, precomp_coeffs_3, precomp_coeffs_4);
        }
    }
}

// compute_output_thread_4: Uses density_4.
void compute_output_thread_4(Eigen::MatrixXcd &A, Eigen::MatrixXcd &B,
                             double precomp_coeffs_1[5][5], double precomp_coeffs_2[5][5],
                             double precomp_coeffs_3[5][5], double precomp_coeffs_4[5][5],
                             double precomp_coeffs_5[5][5],
                             size_t idx) {
    size_t start = (height / num_threads) * idx;
    size_t end = (height / num_threads) * (idx + 1);
    for (size_t i = start; i < end; i++) {
        for (size_t j = 0; j < width; j++) {
            output[i * width + j] = density_4(A, B,
                dr_scale * ((j * 2.0 / width) - 1.0),
                -dr_scale * ((i * 2.0 / height) - 1.0),
                precomp_coeffs_1, precomp_coeffs_2, precomp_coeffs_3,
                precomp_coeffs_4, precomp_coeffs_5);
        }
    }
}

//------------------------------------------------------------------------------
// density_4: Computes density using five sets of precomputed 5x5 coefficient arrays.
double density_4(Eigen::MatrixXcd &A, Eigen::MatrixXcd &B, double a, double b,
                 double precomp_coeffs_1[5][5], double precomp_coeffs_2[5][5],
                 double precomp_coeffs_3[5][5], double precomp_coeffs_4[5][5],
                 double precomp_coeffs_5[5][5]) {
    double values[5] = {0, 0, 0, 0, 0};
    double a_powers[5] = {1, a, a * a, 0, 0};
    double b_powers[5] = {1, b, b * b, 0, 0};
    for (size_t i = 3; i < 5; i++) {
        a_powers[i] = a_powers[i - 1] * a;
        b_powers[i] = b_powers[i - 1] * b;
    }
    for (size_t i = 0; i < 5; i++) {
        for (size_t j = 0; j < 5; j++) {
            double prod = a_powers[i] * b_powers[j];
            values[0] += precomp_coeffs_1[i][j] * prod;
            values[1] += precomp_coeffs_2[i][j] * prod;
            values[2] += precomp_coeffs_3[i][j] * prod;
            values[3] += precomp_coeffs_4[i][j] * prod;
            values[4] += precomp_coeffs_5[i][j] * prod;
        }
    }
    auto roots = quartic_solver(values[4], values[3], values[2], values[1], values[0]);
    double value = 0;
    for (const auto &root : roots) {
        if (root.imag() > eps) {
            value += root.imag() / ((a * root + b) * std::conj(a * root + b)).real();
        }
    }
    return value;
}

//------------------------------------------------------------------------------
// compute_output: Computes the output using matrix C.
void compute_output(Eigen::MatrixXcd &C) {
    Eigen::MatrixXcd A = (C + C.adjoint()) / 2.0;
    Eigen::MatrixXcd B = -If * (C - C.adjoint()) / 2.0;
    
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    if (d == 2) {
        double A_tr = A.trace().real();
        double AA_tr = (A * A).trace().real();
        double B_tr = B.trace().real();
        double BB_tr = (B * B).trace().real();
        double AB_tr = (A * B).trace().real();
        double precomp_coeffs[9] = {
            -AB_tr*AB_tr + 2*A_tr*B_tr*AB_tr - A_tr*A_tr*BB_tr - B_tr*B_tr*AA_tr + AA_tr*BB_tr,
            2*AA_tr*B_tr - 2*A_tr*AB_tr,
            A_tr*A_tr - 2*AA_tr,
            2*BB_tr*A_tr - 2*B_tr*AB_tr,
            4*AB_tr - 2*A_tr*B_tr,
            B_tr*B_tr - 2*BB_tr,
            A_tr*A_tr - AA_tr,
            2*AB_tr - 2*A_tr*B_tr,
            B_tr*B_tr - BB_tr
        };
        for (size_t idx = 0; idx < num_threads; idx++) {
            threads.push_back(std::thread(compute_output_thread_2, std::ref(A), std::ref(B), precomp_coeffs, idx));
        }
    } else if (d == 3) {
        double tr[4][4];
        Eigen::MatrixXcd A_powers[4] = {
            Eigen::MatrixXcd::Identity(A.rows(), A.cols()),
            A, A * A,
            Eigen::MatrixXcd::Identity(A.rows(), A.cols())
        };
        Eigen::MatrixXcd B_powers[4] = {
            Eigen::MatrixXcd::Identity(B.rows(), B.cols()),
            B, B * B,
            Eigen::MatrixXcd::Identity(B.rows(), B.cols())
        };
        A_powers[3] = A_powers[2] * A;
        B_powers[3] = B_powers[2] * B;
        for (size_t i = 0; i < 4; i++) {
            for (size_t j = 0; j < 4; j++) {
                tr[i][j] = (A_powers[i] * B_powers[j]).trace().real();
            }
        }
        double precomp_coeffs_1[7][7] = {
            {0, 0, 0, -(tr[1][0]*tr[1][0]*tr[1][0]) + 3*tr[1][0]*tr[2][0] - 2*tr[3][0],
             3*(tr[1][0]*tr[1][0]) - 3*tr[2][0], -6*tr[1][0], 6},
            {0, 0, -3*tr[0][1]*(tr[1][0]*tr[1][0]) + 6*tr[1][0]*tr[1][1] + 3*tr[0][1]*tr[2][0] - 6*tr[2][1],
             6*tr[0][1]*tr[1][0] - 6*tr[1][1], -6*tr[0][1], 0, 0},
            {0, -3*(tr[0][1]*tr[0][1])*tr[1][0] + 3*tr[0][2]*tr[1][0] + 6*tr[0][1]*tr[1][1] - 6*tr[1][2],
             3*(tr[0][1]*tr[0][1]) - 3*tr[0][2] + 3*(tr[1][0]*tr[1][0]) - 3*tr[2][0],
             -12*tr[1][0], 18, 0, 0},
            {-(tr[0][1]*tr[0][1]*tr[0][1]) + 3*tr[0][1]*tr[0][2] - 2*tr[0][3],
             6*tr[0][1]*tr[1][0] - 6*tr[1][1], -12*tr[0][1], 0, 0, 0, 0},
            {3*(tr[0][1]*tr[0][1]) - 3*tr[0][2], -6*tr[1][0], 18, 0, 0, 0, 0},
            {-6*tr[0][1], 0, 0, 0, 0, 0, 0},
            {6, 0, 0, 0, 0, 0, 0}
        };
        double precomp_coeffs_2[7][7] = {
            {0, 0, 0, -3*tr[0][1]*tr[1][0]*tr[1][0] + 6*tr[1][0]*tr[1][1] + 3*tr[0][1]*tr[2][0] - 6*tr[2][1],
             6*tr[0][1]*tr[1][0] - 6*tr[1][1], -6*tr[0][1], 0},
            {0, 0, -6*tr[0][1]*tr[0][1]*tr[1][0] + 6*tr[0][2]*tr[1][0] + 3*tr[1][0]*tr[1][0]*tr[1][0] +
             12*tr[0][1]*tr[1][1] - 12*tr[1][2] - 9*tr[1][0]*tr[2][0] + 6*tr[3][0],
             6*tr[0][1]*tr[0][1] - 6*tr[0][2] - 6*tr[1][0]*tr[1][0] + 6*tr[2][0], 6*tr[1][0], 0, 0},
            {0, -3*tr[0][1]*tr[0][1]*tr[0][1] + 9*tr[0][1]*tr[0][2] - 6*tr[0][3] +
             6*tr[0][1]*tr[1][0]*tr[1][0] - 12*tr[1][0]*tr[1][1] - 6*tr[0][1]*tr[2][0] + 12*tr[2][1],
             0, -12*tr[0][1], 0, 0, 0},
            {3*tr[0][1]*tr[0][1]*tr[1][0] - 3*tr[0][2]*tr[1][0] - 6*tr[0][1]*tr[1][1] + 6*tr[1][2],
             6*tr[0][1]*tr[0][1] - 6*tr[0][2] - 6*tr[1][0]*tr[1][0] + 6*tr[2][0],
             12*tr[1][0], 0, 0, 0, 0},
            {-6*tr[0][1]*tr[1][0] + 6*tr[1][1], -6*tr[0][1], 0, 0, 0, 0, 0},
            {6*tr[1][0], 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0}
        };
        double precomp_coeffs_3[7][7] = {
            {0, 0, 0, -3*tr[0][1]*tr[0][1]*tr[1][0] + 3*tr[0][2]*tr[1][0] + 6*tr[0][1]*tr[1][1] - 6*tr[1][2],
             3*tr[0][1]*tr[0][1] - 3*tr[0][2], 0, 0},
            {0, 0, -3*tr[0][1]*tr[0][1]*tr[0][1] + 9*tr[0][1]*tr[0][2] - 6*tr[0][3] +
             6*tr[0][1]*tr[1][0]*tr[1][0] - 12*tr[1][0]*tr[1][1] - 6*tr[0][1]*tr[2][0] + 12*tr[2][1],
             -6*tr[0][1]*tr[1][0] + 6*tr[1][1], 0, 0, 0},
            {0, 6*tr[0][1]*tr[0][1]*tr[1][0] - 6*tr[0][2]*tr[1][0] - 3*tr[1][0]*tr[1][0]*tr[1][0] -
             12*tr[0][1]*tr[1][1] + 12*tr[1][2] + 9*tr[1][0]*tr[2][0] - 6*tr[3][0],
             3*tr[0][1]*tr[0][1] - 3*tr[0][2] + 3*tr[1][0]*tr[1][0] - 3*tr[2][0],
             0, 0, 0, 0},
            {-3*tr[0][1]*tr[1][0]*tr[1][0] + 6*tr[1][0]*tr[1][1] + 3*tr[0][1]*tr[2][0] - 6*tr[2][1],
             -6*tr[0][1]*tr[1][0] + 6*tr[1][1], 0, 0, 0, 0, 0},
            {3*tr[1][0]*tr[1][0] - 3*tr[2][0], 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0}
        };
        double precomp_coeffs_4[7][7] = {
            {0, 0, 0, -(tr[0][1]*tr[0][1]*tr[0][1]) + 3*tr[0][1]*tr[0][2] - 2*tr[0][3], 0, 0, 0},
            {0, 0, 3*tr[0][1]*tr[0][1]*tr[1][0] - 3*tr[0][2]*tr[1][0] - 6*tr[0][1]*tr[1][1] + 6*tr[1][2],
             0, 0, 0, 0},
            {0, -3*tr[0][1]*tr[1][0]*tr[1][0] + 6*tr[1][0]*tr[1][1] + 3*tr[0][1]*tr[2][0] - 6*tr[2][1],
             0, 0, 0, 0},
            {tr[1][0]*tr[1][0]*tr[1][0] - 3*tr[1][0]*tr[2][0] + 2*tr[3][0],
             0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0}
        };
        for (size_t idx = 0; idx < num_threads; idx++) {
            threads.push_back(std::thread(compute_output_thread_3, std::ref(A), std::ref(B),
                                           precomp_coeffs_1, precomp_coeffs_2, precomp_coeffs_3, precomp_coeffs_4, idx));
        }
    } else if (d == 4) {
        double tt0 = A.trace().real();
        double tt1 = B.trace().real();
        double tt00 = (A * A).trace().real();
        double tt01 = (A * B).trace().real();
        double tt11 = (B * B).trace().real();
        double tt000 = (A * A * A).trace().real();
        double tt001 = (A * A * B).trace().real();
        double tt011 = (A * B * B).trace().real();
        double tt111 = (B * B * B).trace().real();
        double tt0000 = (A * A * A * A).trace().real();
        double tt0001 = (A * A * A * B).trace().real();
        double tt0011 = (A * A * B * B).trace().real();
        double tt0101 = (A * B * A * B).trace().real();
        double tt0111 = (A * B * B * B).trace().real();
        double tt1111 = (B * B * B * B).trace().real();
        double precomp0[5][5] = {
            {tt1*tt1*tt1*tt1 - 6*tt1*tt1*tt11 + 3*tt11*tt11 + 8*tt1*tt111 - 6*tt1111,
             -4*tt1*tt1*tt1 + 12*tt1*tt11 - 8*tt111,
             12*tt1*tt1 - 12*tt11, -24*tt1, 24},
            {0,0,0,0,0},
            {0,0,0,0,0},
            {0,0,0,0,0},
            {0,0,0,0,0}
        };
        double precomp1[5][5] = {
            {-24*tt0111 + 24*tt011*tt1 - 12*tt01*tt1*tt1 + 4*tt0*tt1*tt1*tt1 + 12*tt01*tt11 - 12*tt0*tt1*tt11 + 8*tt0*tt111,
             -24*tt011 + 24*tt01*tt1 - 12*tt0*tt1*tt1 + 12*tt0*tt11,
             -24*tt01 + 24*tt0*tt1, -24*tt0, 0},
            {-4*tt1*tt1*tt1 + 12*tt1*tt11 - 8*tt111,
             24*tt1*tt1 - 24*tt11, -72*tt1, 96, 0},
            {0,0,0,0,0},
            {0,0,0,0,0},
            {0,0,0,0,0}
        };
        double precomp2[5][5] = {
            {-24*tt0011 + 12*tt01*tt01 - 12*tt0101 + 24*tt0*tt011 + 24*tt001*tt1 - 24*tt0*tt01*tt1 +
             6*tt0*tt0*tt1*tt1 - 6*tt00*tt1*tt1 - 6*tt0*tt0*tt11 + 6*tt00*tt11,
             -24*tt001 + 24*tt0*tt01 - 12*tt0*tt0*tt1 + 12*tt00*tt1,
             12*tt0*tt0 - 12*tt00, 0, 0},
            {-24*tt011 + 24*tt01*tt1 - 12*tt0*tt1*tt1 + 12*tt0*tt11,
             -48*tt01 + 48*tt0*tt1, -72*tt0, 0, 0},
            {12*tt1*tt1 - 12*tt11, -72*tt1, 144, 0, 0},
            {0,0,0,0,0},
            {0,0,0,0,0}
        };
        double precomp3[5][5] = {
            {-24*tt0001 + 24*tt0*tt001 - 12*tt0*tt0*tt01 + 12*tt00*tt01 + 4*tt0*tt0*tt0*tt1 -
             12*tt0*tt00*tt1 + 8*tt000*tt1, -4*tt0*tt0*tt0 + 12*tt0*tt00 - 8*tt000,
             0, 0, 0},
            {-24*tt001 + 24*tt0*tt01 - 12*tt0*tt0*tt1 + 12*tt00*tt1,
             24*tt0*tt0 - 24*tt00, 0, 0, 0},
            {-24*tt01 + 24*tt0*tt1, -72*tt0, 0, 0, 0},
            {-24*tt1, 96, 0, 0, 0},
            {0,0,0,0,0}
        };
        double precomp4[5][5] = {
            {tt0*tt0*tt0*tt0 - 6*tt0*tt0*tt00 + 3*tt00*tt00 + 8*tt0*tt000 - 6*tt0000,
             0, 0, 0, 0},
            {-4*tt0*tt0*tt0 + 12*tt0*tt00 - 8*tt000,
             0, 0, 0, 0},
            {12*tt0*tt0 - 12*tt00, 0, 0, 0, 0},
            {-24*tt0, 0, 0, 0, 0},
            {24, 0, 0, 0, 0}
        };
        for (size_t idx = 0; idx < num_threads; idx++) {
            threads.push_back(std::thread(compute_output_thread_4, std::ref(A), std::ref(B),
                                           precomp0, precomp1, precomp2, precomp3, precomp4, idx));
        }
    } else {
        for (size_t idx = 0; idx < num_threads; idx++) {
            threads.push_back(std::thread(compute_output_thread, std::ref(A), std::ref(B), idx));
        }
    }
    
    for (auto &t : threads) {
        t.join();
    }
    
    // Process the raw output into output_buffer.
    for (size_t i = 0; i < height; i++) {
        for (size_t j = 0; j < width; j++) {
            double comp = 40.0 * output[i * width + j] / (1 + 40.0 * output[i * width + j]);
            uint32_t comp_255 = static_cast<uint32_t>(static_cast<uint8_t>(comp * 255.0));
            output_buffer[i * width + j] = (0x000000FF |
                ((255 - comp_255) << 8) | ((255 - comp_255) << 16) | (comp_255 << 24)) ^ (comp_255 >> 1);
        }
    }
}

//------------------------------------------------------------------------------
// set_zoom: Adjusts the zoom factor.
void set_zoom(double zoom) {
    zoom_scale = zoom;
}

//------------------------------------------------------------------------------
// set_matrix: Converts a given pointer to a complex matrix and computes the output.
void set_matrix(size_t dim, size_t pointer) {
    d = static_cast<int>(dim);
    dr_scale = zoom_scale * dim * p_scale;
    double *entries = reinterpret_cast<double*>(pointer);
    Eigen::MatrixXcd C = Eigen::MatrixXcd::Zero(d, d);
    for (size_t i = 0; i < static_cast<size_t>(d); i++) {
        for (size_t j = 0; j < static_cast<size_t>(d); j++) {
            C(i, j) = entries[2 * (i * d + j)] +
                      entries[2 * (i * d + j) + 1] * If;
        }
    }
    compute_output(C);
}

//------------------------------------------------------------------------------
// set_resolution: Configures resolution and allocates working memory.
void set_resolution(size_t resolution) {
    if (output) { delete [] output; }
    if (output_buffer) { delete [] output_buffer; }
    width = height = resolution;
    output = new double[width * height];
    output_buffer = new uint32_t[width * height];
    std::memset(output, 0, width * height * sizeof(double));
    num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) {
        num_threads = 1;
    }
}

#ifndef BUILD_PYTHON
// Simple main() for the standalone executable.
int main() {
    std::cout << "spectral_density_exec: Standalone spectral density computation." << std::endl;
    
    // Set a resolution.
    set_resolution(256);
    
    // Create a dummy 2x2 complex matrix.
    // Format: [real0, imag0, real1, imag1, real2, imag2, real3, imag3]
    double dummy_data[] = { 1.0, 0.0,
                            2.0, 0.0,
                            3.0, 0.0,
                            4.0, 0.0 };
                            
    // Compute the output using the dummy matrix.
    set_matrix(2, reinterpret_cast<size_t>(dummy_data));
    
    std::cout << "Computation complete. First output pixel: " << output_buffer[0] << std::endl;
    return 0;
}
#endif
