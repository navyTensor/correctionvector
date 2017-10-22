#include "itensor/all_basic.h"
#include "../cc/gmres.h"

using namespace itensor;

Vector
VectorProductFnc(Matrix const& A,
                 Vector const& x)
{
    return A*x;
}

Real
VectorInnerFnc(Vector const& a,
               Vector const& b)
{
    return a*b;
}

ITensor
TensorProductFnc(ITensor const& A,
                 ITensor const& x)
{
    return dag(noprime(A*x));
}

Real
TensorInnerFnc(ITensor const& a,
               ITensor const& b)
{
    return (dag(a) * b).real();
}

int main()
{
    // Test to see if can solve a simple matrix
    Matrix A(3,3);
    A(0,0) = 3;   A(0,1) = 2;   A(0,2) = -1;
    A(1,0) = 2;   A(1,1) = -2;  A(1,2) = 4;
    A(2,0) = -1;  A(2,1) = 0.5; A(2,2) = -1;
    Vector b(3);
    b(0) = 1; b(1) = -2; b(2) = 0;
    Vector x(3);
    for(auto& el : x) el = 0.0;

    Vector soln(3);
    soln(0) = 1; soln(1) = -2; soln(2) = -2;

    int maxiter;
    double threshold;
    gmres<Vector,Matrix>(A, b, x, VectorProductFnc, VectorInnerFnc, threshold, maxiter);
   
    int success = 0;
    for(auto i : range(soln.size()))
    {
        if(std::norm( (x(i)-soln(i)) > 1E-10)) success = 1;
    }
    if(success != 0) std::cout << "FAIL: GMRES solving a real basic matrix." << std::endl;
    else std::cout << "PASS: GMRES solved a real basic matrix." << std::endl;
    
    // Now test with tensors
    // Create the RHS
    auto s = Index("s",3);
    auto psib = ITensor(s);
    psib.set(s(1), 1); psib.set(s(2), -2); psib.set(s(3), 0);

    // Create the A matrix
    auto H = ITensor(s,prime(s));
    H.set(s(1),prime(s)(1), 3);    H.set(s(2),prime(s)(1), 2);    H.set(s(3),prime(s)(1), -1);
    H.set(s(1),prime(s)(2), 2);    H.set(s(2),prime(s)(2), -2);   H.set(s(3),prime(s)(2), 4);
    H.set(s(1),prime(s)(3), -1);   H.set(s(2),prime(s)(3), 0.5);  H.set(s(3),prime(s)(3), -1);

    // random initial guess
    auto psi = ITensor(s);
    psi.set(s(1), rand() % 10); psi.set(s(2), rand() % 10); psi.set(s(3), rand() % 10);
    
    // THe expected soln
    auto psisoln = ITensor(s);
    psisoln.set(s(1), 1); psisoln.set(s(2), -2); psisoln.set(s(3), -2);

    int maxiter2;
    double threshold2;
    gmres<Vector,Matrix>(H, psib, psi, TensorProductFnc, TensorInnerFnc, threshold2, maxiter2);

    int success2 = 0;
    for(auto i : range(3))
    {
        if(std::norm( (psi.real(s(i+1))-psisoln.real(s(i+1))) > 1E-10)) success2 = 1;
    }
    if(success2 != 0) std::cout << "FAIL: GMRES solving a real basic tensor." << std::endl;
    else std::cout << "PASS: GMRES solved a real basic tensor." << std::endl;
    
    return 0;
}
