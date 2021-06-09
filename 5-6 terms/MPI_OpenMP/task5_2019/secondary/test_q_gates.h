#include <complex>

using namespace std;
using complexd = complex<double>;

complexd* test_single_q_t(complexd* a, size_t n, 
                    complexd u[2][2], size_t k);

complexd* test_double_q_t(complexd* a, size_t n,
                    complexd u[4][4], size_t k, size_t l);

complexd* test_adamar_gate(complexd* a, size_t n, size_t k);

complexd* test_n_adamar_gate(complexd* a, size_t n);

complexd* test_rw_gate(complexd* a, size_t n, size_t k, complexd phi);

complexd* test_c_rw_gate(complexd* a, size_t n, size_t k,
                    size_t l, complexd phi);

complexd* test_not_gate(complexd* a, size_t n, size_t k);

complexd* test_c_not_gate(complexd* a, size_t n, size_t k,
                size_t l);

complexd* test_swap_gate(complexd* a, size_t n, size_t k,
                size_t l);