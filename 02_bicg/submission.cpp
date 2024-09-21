#include <math.h>
#include <omp.h>
#include <vector>

CRSMatrix transposeCRS(const CRSMatrix &A) {
    std::vector<std::vector<int>> intVectors(A.n);
    std::vector<std::vector<double>> dblVectors(A.n);

    for (int i = 0; i < A.n; i++) {
        intVectors[i].reserve(A.nz / A.n);
        dblVectors[i].reserve(A.nz / A.n);
    }

    for (int i = 0; i < A.n; i++) {
        for (int j = A.rowPtr[i]; j < A.rowPtr[i + 1]; j++) {
            int col = A.colIndex[j];
            intVectors[col].push_back(i);
            dblVectors[col].push_back(A.val[j]);
        }
    }

    CRSMatrix AT;
    AT.n = A.n;
    AT.m = A.n;
    AT.nz = 0;

    AT.rowPtr.reserve(A.n + 1);
    size_t valCount = 0;
    for (int i = 0; i < A.n; i++) {
        valCount += dblVectors[i].size();
    }
    AT.val.reserve(valCount);
    AT.colIndex.reserve(valCount);

    for (int i = 0; i < A.n; i++) {
        AT.rowPtr.push_back(AT.nz);
        for (int j = 0; j < dblVectors[i].size(); j++) {
            AT.val.push_back(dblVectors[i].at(j));
            AT.colIndex.push_back(intVectors[i].at(j));
            AT.nz++;
        }
    }
    AT.rowPtr.push_back(AT.nz);

    return AT;
}

void multCRS(CRSMatrix &A, const double *x, double *result) {
#pragma omp parallel for
    for (int i = 0; i < A.n; i++) {
        result[i] = 0.0;
        for (int j = A.rowPtr[i]; j < A.rowPtr[i + 1]; j++) {
            result[i] += A.val[j] * x[A.colIndex[j]];
        }
    }
}

double scalarProduct(const double *a, const double *b, int n) {
    double result = 0;
    for (int i = 0; i < n; i++) {
        result += a[i] * b[i];
    }

    return result;
}

void addMultVector(double *x, double multFactor1, const double *addition, double multFactor2, int n) {
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        x[i] = x[i] * multFactor1 + addition[i] * multFactor2;
    }
}

double l2Norm(const double *x, int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += x[i] * x[i];
    }
    return sqrt(sum);
}

void subtract(const double *x, const double *y, double *res, int n) {
    for (int i = 0; i < n; i++) {
        res[i] = x[i] - y[i];
    }
}

void SLE_Solver_CRS_BICG(CRSMatrix &A, double *b, double eps, int max_iter, double *x, int &count) {
    CRSMatrix AT = transposeCRS(A);
    double *tmp = new double[A.n];
    double *r = new double[A.n];
    double *p = new double[A.n];
    double *z = new double[A.n];
    double *s = new double[A.n];
    for (int i = 0; i < A.n; i++) {
        x[i] = tmp[i] = 0.;
        r[i] = p[i] = z[i] = s[i] = b[i];
    }

    double alpha_k;
    double beta_k;
    for (count = 0; count < max_iter; count++) {
        multCRS(A, z, tmp);
        double prProd = scalarProduct(p, r, A.n);
        alpha_k = prProd / scalarProduct(s, tmp, A.n);

        addMultVector(x, 1., z, alpha_k, A.n);
        addMultVector(r, 1., tmp, -alpha_k, A.n);

        multCRS(AT, s, tmp);
        addMultVector(p, 1., tmp, -alpha_k, A.n);

        beta_k = scalarProduct(p, r, A.n) / prProd;

        addMultVector(z, beta_k, r, 1., A.n);
        addMultVector(s, beta_k, p, 1., A.n);

        if (abs(beta_k) < 1e-14 || l2Norm(r, A.n) < eps)
            break;
    }
}
