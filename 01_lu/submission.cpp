#include <omp.h>

constexpr int estimatedBlockSize = 200;

void UfillZero(double *U, const int size) {
#pragma omp parallel for
  for (int i = 1; i < size; i++) {
    for (int j = 0; j < i; j++) {
      U[i * size + j] = 0;
    }
  }
}

void LfillZero(double *L, const int size) {
#pragma omp parallel for
  for (int i = 0; i < size; i++) {
    for (int k = i + 1; k < size; k++) {
      L[i * size + k] = 0;
    }
  }
}

void createNewBlock(double *A, double *L, double *U, int N, int blockSize,
                    int shift) {
  int KblockSize;
  int IblockSize;
  int JblockSize;

#pragma omp parallel private(KblockSize, IblockSize, JblockSize)
  {
    int KblockSize = 50;
    int IblockSize = 50;
    int JblockSize = 50;
#pragma omp for
    for (int iBlIt = shift + blockSize; iBlIt < N; iBlIt += IblockSize) {
      int endOfBlockI = iBlIt + IblockSize;
      if (endOfBlockI > N)
        IblockSize = N - iBlIt;
      for (int jBlIt = shift + blockSize; jBlIt < N; jBlIt += JblockSize) {
        int endOfBlockJ = jBlIt + JblockSize;
        if (endOfBlockJ > N)
          JblockSize = N - jBlIt;
        for (int kBlIt = shift; kBlIt < shift + blockSize;
             kBlIt += KblockSize) {
          for (int i = iBlIt; i < iBlIt + IblockSize; i++) {
            for (int j = jBlIt; j < jBlIt + JblockSize; j++) {
              for (int k = kBlIt; k < kBlIt + KblockSize; k++) {
                A[i * N + j] = A[i * N + j] - L[i * N + k] * U[k * N + j];
              }
            }
          }
        }
      }
    }
  }
}

void luBlocked(double *A, double *L, double *U, size_t N, size_t shift,
               size_t blockSize) {

  for (int i = shift; i < shift + blockSize; i++) {
#pragma omp parallel for
    for (int j = shift; j < shift + blockSize; j++) {
      U[N * i + j] = A[N * i + j];
    }
  }

  for (int i = shift; i < shift + blockSize; i++) {

    L[i * N + i] = 1;

#pragma omp parallel for
    for (int k = i + 1; k < shift + blockSize; k++) {

      double multiplier = U[k * N + i] / U[i * N + i];
      for (int j = i; j < shift + blockSize; j++) {

        U[k * N + j] = U[k * N + j] - multiplier * U[i * N + j];
      }

      L[k * N + i] = multiplier;
    }
  }
}

void LU_Decomposition(double *A, double *L, double *U, int N) {
  size_t needleBlockSize = std::min(N, estimatedBlockSize);
  std::memset(L, 0, sizeof(double) * N * N);
  size_t blocksNum = N / needleBlockSize;
  std::memset(U, 0, sizeof(double) * N * N);
  for (size_t i = 0; i < blocksNum; i++) {
    size_t shift = needleBlockSize * i;
    if (N < shift + 2 * needleBlockSize)
      needleBlockSize = N - shift;

    luBlocked(A, L, U, N, shift, needleBlockSize);

    if (N <= shift + needleBlockSize)
      break;

#pragma omp parallel for
    for (int colShift = shift + needleBlockSize; colShift < N; colShift++) {
      for (int i = shift; i < shift + needleBlockSize; i++) {
        U[i * N + colShift] = A[i * N + colShift] / L[i * N + i];
        for (int j = i + 1; j < shift + needleBlockSize; j++) {
          A[j * N + colShift] =
              A[j * N + colShift] - L[j * N + i] * U[i * N + colShift];
        }
      }
    }

#pragma omp parallel for
    for (int rowShift = shift + needleBlockSize; rowShift < N; rowShift++) {
      for (int i = shift; i < shift + needleBlockSize; i++) {
        L[rowShift * N + i] = A[rowShift * N + i] / U[i * N + i];
        for (int j = i + 1; j < shift + needleBlockSize; j++) {
          A[rowShift * N + j] =
              A[rowShift * N + j] - U[i * N + j] * L[rowShift * N + i];
        }
      }
    }
    createNewBlock(A, L, U, N, needleBlockSize, shift);
  }
  UfillZero(U, N);
  LfillZero(L, N);
}
