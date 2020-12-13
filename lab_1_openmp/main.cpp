#include <iostream>
#include <omp.h>
#include <iomanip>
#include <cmath>

using namespace std;

#define EPS 0.000001
#define INIT_N 1000000

// ********************************
// **           MATH             **
// ********************************

double f(double x) {
    return pow(sin(1 / x) / x, 2);
}

double integral(double a, double b) {
    return (2 * (b - a) / (a * b) + sin(2 / b) - sin(2 / a)) / 4;
}

// ********************************
// **          METHODS           **
// ********************************

double serial(double a, double b, int n) {
    double step = (b - a) / n;
    double x = 0;
    double sum = 0;

    for (int i = 1; i < n - 1; i++) {
        x = a + step * i;
        sum += f(x);
    }

    return step * (f(a) / 2 + sum + f(b) / 2);
}

double atomic(double a, double b, int threads, int n) {
    double step = (b - a) / n;
    double x = 0;
    double sum = 0;

    #pragma omp parallel for num_threads(threads) private(x)
    for (int i = 1; i < n - 1; i++) {
        x = a + step * i;
        #pragma omp atomic
        sum += f(x);
    }

    return step * (f(a) / 2 + sum + f(b) / 2);
}

double critical(double a, double b, int threads, int n) {
    double step = (b - a) / n;
    double x = 0;
    double sum = 0;

    #pragma omp parallel for num_threads(threads) private(x)
    for (int i = 1; i < n - 1; i++) {
        x = a + step * i;
        #pragma omp critical
        sum += f(x);
    }

    return step * (f(a) / 2 + sum + f(b) / 2);
}

double locks(double a, double b, int threads, int n) {
    double step = (b - a) / n;
    double x = 0;
    double sum = 0;

    omp_lock_t lock;
    omp_init_lock(&lock);

    #pragma omp parallel for num_threads(threads) private(x)
    for (int i = 1; i < n - 1; i++) {
        x = a + step * i;
        omp_set_lock (&lock);
        sum += f(x);
        omp_unset_lock (&lock);
    }

    omp_destroy_lock (&lock);

    return step * (f(a) / 2 + sum + f(b) / 2);
}

double reduction(double a, double b, int threads, int n) {
    double step = (b - a) / n;
    double x = 0;
    double sum = 0;

    #pragma omp parallel num_threads(threads) private(x)
    {
        #pragma omp for reduction(+:sum)
        for (int i = 1; i < n - 1; i++) {
            x = a + step * i;
            sum += f(x);
        }
    }

    return step * (f(a) / 2 + sum + f(b) / 2);
}

// ********************************
// **          HELPERS           **
// ********************************

int getN(double a, double b, double eps, int initN) {
    double x = integral(a, b);

    int n = initN;

    while (abs(x - serial(a, b, n)) > eps * x) {
        n += 1;
    }

    return n;
}

void runSerial(int n[]) {
    int i = 0;
    for (double a = 0.00001, b = 0.0001; a <= 10 && b <= 100; a *= 10, b *= 10, i += 1) {
        double t = omp_get_wtime();
        serial(a, b, n[i]);
        double time = omp_get_wtime() - t;
        cout << time << endl;
    }
}

void runAtomic(int threads, int n[]) {
    int i = 0;
    for (double a = 0.00001, b = 0.0001; a <= 10 && b <= 100; a *= 10, b *= 10, i += 1) {
        double t = omp_get_wtime();
        atomic(a, b, threads, n[i]);
        double time = omp_get_wtime() - t;
        cout << time << endl;
    }
}

void runCritical(int threads, int n[]) {
    int i = 0;
    for (double a = 0.00001, b = 0.0001; a <= 10 && b <= 100; a *= 10, b *= 10, i += 1) {
        double t = omp_get_wtime();
        critical(a, b, threads, n[i]);
        double time = omp_get_wtime() - t;
        cout <<  time << endl;
    }
}

void runLocks(int threads, int n[]) {
    int i = 0;
    for (double a = 0.00001, b = 0.0001; a <= 10 && b <= 100; a *= 10, b *= 10, i += 1) {
        double t = omp_get_wtime();
        locks(a, b, threads, n[i]);
        double time = omp_get_wtime() - t;
        cout <<  time << endl;
    }
}

void runReduction(int threads, int n[]) {
    int i = 0;
    for (double a = 0.00001, b = 0.0001; a <= 10 && b <= 100; a *= 10, b *= 10, i += 1) {
        double t = omp_get_wtime();
        reduction(a, b, threads, n[i]);
        double time = omp_get_wtime() - t;
        cout <<  time << endl;
    }
}

// ********************************
// **           MAIN             **
// ********************************

int main() {
    int n[7] = {
            getN(0.00001, 0.0001, EPS, INIT_N),
            getN(0.0001, 0.001, EPS, INIT_N),
            getN(0.001, 0.01, EPS, INIT_N),
            getN(0.01, 0.1, EPS, INIT_N),
            getN(0.1, 1, EPS, INIT_N),
            getN(1, 10, EPS, INIT_N),
            getN(10, 100, EPS, INIT_N)
    };

    cout << "serial: " << endl;
    runSerial(n);

    for (int threads = 1; threads <= 16; threads *= 2) {
        cout << "threads: " << threads << endl;
        cout << "atomic:" << endl;
        runAtomic(threads, n);
        cout << "critical:" << endl;
        runCritical(threads, n);
        cout << "locks:" << endl;
        runLocks(threads, n);
        cout << "reduction:" << endl;
        runReduction(threads, n);
    }
}
