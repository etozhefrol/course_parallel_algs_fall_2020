#include "mpi.h"
#include <omp.h>
#include <iostream>

using namespace std;

#define SIZE 1000

int a[SIZE][SIZE];
int b[SIZE][SIZE];

//int a[SIZE][SIZE] = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};
//int b[SIZE][SIZE] = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};

int cSerial[SIZE][SIZE];
int cPar1[SIZE][SIZE];
int cPar2[SIZE][SIZE];

void runSerial(int argc, char* argv[]) {
    int procRank, procNum;

    MPI_Status status;
//    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &procNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);

    if (procRank == 0) {
        double t1 = MPI_Wtime();

        // подсчёт произведений по строкам
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                cSerial[i][j] = 0;
                for (int k = 0; k < SIZE; k++) {
                    cSerial[i][j] += a[i][k] * b[k][j];
                }
            }
        }

        // вывод
//        for (int i = 0; i < SIZE; i++) {
//            for (int j = 0; j < SIZE; j++) {
//                cout << cSerial[i][j] << " ";
//            }
//            cout << endl;
//        }

        cout << MPI_Wtime() - t1 << endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

void runParallel(int argc, char* argv[]) {
    int procRank, procNum;
    int procA[SIZE][SIZE];
    int procRowsOffset = 0;
    double t1;

    MPI_Status status;
//    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &procNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);

    int workersNum = procNum - 1;
    int procRowsNum = SIZE / workersNum;

    // отправка и получение данных для произведения
    if (procRank == 0) {
        t1 = MPI_Wtime();

        for (int i = 1; i < procNum; i++, procRowsOffset += procRowsNum) {
            MPI_Send(&procRowsOffset, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&a[procRowsOffset][0], procRowsNum * SIZE, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(&procRowsOffset, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(&procA, procRowsNum * SIZE, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    }

    // подсчёт произведений по строкам
    if (procRank != 0) {
        for (int i = 0; i < procRowsNum; i++) {
            for (int j = 0; j < SIZE; j++) {
                cPar1[i][j] = 0;
                for (int k = 0; k < SIZE; k++) {
                    cPar1[i][j] += procA[i][k] * b[k][j];
                }
            }
        }
    }

    // отправка и получение результатов
    if (procRank != 0) {
        MPI_Send(&procRowsOffset, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&cPar1, procRowsNum * SIZE, MPI_INT, 0, 0, MPI_COMM_WORLD);
    } else {
        for (int i = 1; i < procNum; i++) {
            MPI_Recv(&procRowsOffset, 1, MPI_INT, i, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            MPI_Recv(&cPar1[procRowsOffset][0], procRowsNum * SIZE, MPI_INT, i, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        }
    }

    // вывод
    if (procRank == 0) {
        cout << MPI_Wtime() - t1 << endl;

//        for (int i = 0; i < SIZE; i++) {
//            for (int j = 0; j < SIZE; j++) {
//                cout << cPar1[i][j] << " ";
//            }
//            cout << endl;
//        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

void runParallel2(int argc, char* argv[]) {
    int procRank, procNum;
    int procColsOffset = 0;
    int procB[SIZE][SIZE];
    double t1;

    MPI_Status status;
//    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &procNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);

    int workersNum = procNum - 1;
    int procColsNum = SIZE / workersNum;

    MPI_Datatype coltype;
    MPI_Type_vector(SIZE, procColsNum, SIZE, MPI_INT, &coltype);
    MPI_Type_commit(&coltype);


    // отправка и получение данных для произведения
    if (procRank == 0) {
        t1 = MPI_Wtime();

        for (int i = 1; i < procNum; i++, procColsOffset += procColsNum) {
            MPI_Send(&procColsOffset, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&b[0][procColsOffset], 1, coltype, i, 0, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(&procColsOffset, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(&procB, 1, coltype, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    }

    // подсчёт произведений по строкам
    if (procRank != 0) {
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < procColsNum; j++) {
                cPar2[i][j + procColsOffset] = 0;
                for (int k = 0; k < SIZE; k++) {
                    cPar2[i][j + procColsOffset] += a[i][k] * procB[k][j];
                }
            }
        }
    }

    // отправка и получение результатов
    if (procRank != 0) {
        MPI_Send(&procColsOffset, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&cPar2[0][procColsOffset], 1, coltype, 0, 0, MPI_COMM_WORLD);
    } else {
        for (int i = 1; i < procNum; i++) {
            MPI_Recv(&procColsOffset, 1, MPI_INT, i, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            MPI_Recv(&cPar2[0][procColsOffset], 1, coltype, i, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        }
    }

    // вывод
    if (procRank == 0) {
        cout << MPI_Wtime() - t1 << endl;

//        for (int i = 0; i < SIZE; i++) {
//            for (int j = 0; j < SIZE; j++) {
//                cout << cPar2[i][j] << " ";
//            }
//            cout << endl;
//        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char* argv[]) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            a[i][j] = 1 + i * SIZE + j;
            b[i][j] = 1 + i * SIZE + j;
        }
    }

    MPI_Init(&argc, &argv);

//    runSerial(argc, argv);
//    runParallel(argc, argv);
    runParallel2(argc, argv);

    MPI_Finalize();
}