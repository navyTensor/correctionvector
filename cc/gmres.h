/* gmres.h copyleft 2017 hnourse // Adapted from wikipedia and ianmcc MatrixProductToolkit
   
   Use the Generalised Minimal RESidual iteration to solve A x = b.
   
   NOTE: Need to call by specifying a type for Vector and Matrix, which require
         accessing elements as vector(i) and matrix(i,j), and additionally require
         size() methods.
   Example: gmres<myvector,mymatrix>(A,b,x,myproduct,myinner,threshold,max_iter,max_krylov_size)
   NOTE2: x and b (the tensors) need to have an operator += (-=) that adds their objects together.
          The tensors also need to be able to be multiplied by a scalar as tensor * scalar.
   NOTE3: Also require a norm function to inherit from the workspace.
   
   Parameters
   ----------
   A : Linear operator that acts on x
   b : Right hand side of the equation
   x : Initial guess to the solution
   product : Function that implements the action of A on x as product(A,x)
   inner : Function that implements the inner product <a,b> as inner(b,a)
   threshold : Tolerance target to achieve in the residual (error)
   max_iter : Maximum number of iterations regardless of whether threshold is met
   max_krylov_size : Number of iterations between restarts, which effectively sets
       the maximum size of the Krylov space

   Returns
   -------
   Function returns as an int 1 if tolerance is not met within iterations or 0 if successful
   x : Solution to the linear equation
   threshold : Tolerance achieved
   max_iter Number of iterations it took
*/ 

#ifndef __HNOURSE_GMRES_H_
#define __HNOURSE_GMRES_H_

#include <complex>

template <class BigMatrixT, class Tensor, class Matrix, class ProductFnc, class InnerFnc>
void
arnoldi(BigMatrixT const& A, 
        std::vector<Tensor>& Q, 
        Matrix& H,
        ProductFnc product,
        InnerFnc inner,
        int const k)
{
    // Apply another Ax, the nth vector in the Krylov subspace
    auto q = product(A,Q[k]);
    for(int i=0; i<=k; i++)
    {
        H(i,k) = inner(Q[i], q);
        q -= H(i, k) * Q[i];
    }
    H(k+1, k) = norm(q);
    Q[k+1] = q / H(k+1, k);
}

template <class Matrix, class Vector>
void
givens_rotation(Matrix& H, 
                Vector& cs, 
                Vector& sn, 
                int const k)
{
    // Apply for ith column
    for(int i=0; i<k; i++)
    {
        auto temp = cs(i)*H(i,k) + sn(i)*H(i+1,k);
        H(i+1,k) = -sn(i)*H(i,k) + cs(i)*H(i+1,k);
        H(i,k) = temp;
    }
    
    // Do the givens rotation (updates cs(k) and sn(k)
    auto v1 = H(k,k);
    auto v2 = H(k+1,k);
    if(v1 == 0.0)
    {
        cs(k) = 0.0;
        sn(k) = 1.0;
    }
    else
    {
        auto t = std::sqrt(std::norm(v1) + std::norm(v2));  // norm is so complex x+iy returns x*x + y*y
        cs(k) = std::abs(v1) / t;
        sn(k) = cs(k) * v2 / v1;
    }
    
    // Update H and eliminate H(k+1,k)
    H(k,k) = cs(k)*H(k,k) + sn(k)*H(k+1,k);
    H(k+1,k) = 0.0;
}

/* Simple backsolve to invert the matrix (assumes upper triangular and non-singular)
   which the givens rotation guarantees. Returns x.
*/
template<class Tensor, class Matrix, class Vector>
void
backsolve_x(std::vector<Tensor> const& Q, 
        Matrix const& H, 
        Tensor& x, 
        Vector const& resid_vec, 
        int const k)
{
    auto y = resid_vec;      // initialises y with the contents of the norm vector
    
    for(int i=k; i>=0; i--)
    {
        y(i) /= H(i,i);
        for(int j=i-1; j>=0; j--)
        {
            y(j) -= H(j,i) * y(i);
        }
    }
    for(int i=0; i<=k; i++)
    {
        x += Q[i] * y(i);
    }
}

template <class Vector, class Matrix, class BigMatrixT, class Tensor, class ProductFnc, class InnerFnc>
int
gmres(BigMatrixT const& A,
      Tensor const& b,
      Tensor& x,
      ProductFnc product,
      InnerFnc inner,
      double& threshold,
      int& max_iter,
      int const max_krylov_size = 20)
{
    double Approx0 = 1E-15;
    
    if(threshold <= Approx0) threshold = 1E-5;
    if(max_iter < 1) max_iter = 100;//b.size()*10;

    // Initialise the 1D vectors and the upper Hessenberg matrix
    Vector resid_vec(max_krylov_size+1);                            // A vector of the frobenius norms in arnoldi basis
    Vector cs(max_krylov_size+1), sn(max_krylov_size+1);            // Yah I don't really know what these are in the maths
    Matrix H(max_krylov_size+1,max_krylov_size+1);                  // Upper Hessenberg matrix
    // I don't need this
    //for(auto& el : H) el = NAN;                                   // Initialise to NAN in case something bad happens

    // Construct the Q matrix. where each "column" holds a Tensor
    // Note: I would normally construct this from the Vector class, but ITensor doesn't have it's vector templated,
    // it's fixed to be either real (Vector) or complex (CVector). Annoying af tbh, but STL is fine anyway.
    std::vector<Tensor> Q(max_krylov_size+1);
    
    // Store the frobenius norm of the rhs vector (never changes)
    auto b_norm = norm(b);
    
    int i=0, k;
    double residual = -1;
    while(i<=max_iter)
    {
        // Use x as the first guess
        auto r = b - product(A,x);
        auto beta = norm(r);          // store the frobenius norm of the first guess
   
        // Check residual, maybe our guess was already correct
        if(b_norm == Approx0) b_norm = 1.0;
        residual = beta / b_norm;
        if(residual <= threshold)
        {
            threshold = residual;
            max_iter = i;
            return 0;
        }

        // Initialise the first vector in Q
        Q[0] = r / beta;
        for(auto& el : resid_vec) el = 0.0;
        resid_vec(0) = beta;
       
        // Now solve
        for(k=0; k<max_krylov_size && i<=max_iter; k++,i++)
        {
            // Run arnoldi to give orthonormal vectors of our Kyrlov space
            arnoldi(A, Q, H, product, inner, k);

            // Eliminate last element in H in the ith row and update the rotation matrix
            givens_rotation(H, cs, sn, k);

            // Update the residual vector
            resid_vec(k+1) = -sn(k)*resid_vec(k);
            resid_vec(k) = cs(k)*resid_vec(k);
            residual = std::abs(resid_vec(k+1)) / b_norm;

            // Check if we meet tolerance, and if so backsolve for x
            if (residual < threshold) {
                threshold = residual;
                max_iter = k;
                backsolve_x(Q, H, x, resid_vec, k);
                return 0;
            }
        }
        // Restart the Krylov space generation
        backsolve_x(Q, H, x, resid_vec, k-1);
    }
    // It all went wrong and didn't converge
    threshold = residual;
    return 1;
}

#endif
