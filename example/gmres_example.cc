#include "itensor/all_mps.h"
#include "gmres.h"

using namespace itensor;

// These are for the objects similar to eigensolver in a dmrg sweep
ITensor
TensorProductFnc(LocalMPO<ITensor> const& A,
              ITensor const& x)
{
    ITensor y;
    A.product(x,y);
    return y;
}

Complex
TensorInnerFnc(ITensor const& a,
               ITensor const& b)
{
    return (dag(a)*b).cplx();
}

int main()
{
    int maxiter = 100;
    double threshold = 1E-12;
    int max_krylov_size = 20;
    // Call like this
    // Heff is a LocalMPO<> at position "a"
    // psib is the rhs tensor at position "a"
    // psi is the guess tensor (and result) at position "a"
    auto success = gmres<CVector,CMatrix>(Heff, psib, psi, TensorProductFnc, TensorInnerFnc, threshold, maxiter, max_krylov_size);

    return 0;
}
