#include <complex>
#include <cmath>

using namespace std;
using complexd = complex<double>;

complexd* test_single_q_t(complexd* a, size_t n, 
                    complexd u[2][2], size_t k) {
    int vec_size = 1 << n;
    complexd *out = new complexd[vec_size];
	int shift = n - k - 1;
	int pow2q = 1 << shift;

	int N = 1 << n;
	for	(int i = 0; i < N; i++)
	{
		int i0 = i & ~pow2q;
		int i1 = i | pow2q;
		int iq = (i & pow2q) >> shift;
		out[i] = u[iq][0] * a[i0] + u[iq][1] * a[i1];
	}

    return out;
}

complexd* test_double_q_t(complexd* a, size_t n,
                    complexd u[4][4], size_t k, size_t l) {
    int vec_size = 1 << n;
    complexd *out = new complexd[vec_size];
	int shift1 = n - k - 1;
	int shift2 = n - l - 1;
	int pow2q1=1<<(shift1);
	int pow2q2=1<<(shift2);
	int N=1<<n;
	for	(int i=0; i<N; i++)
	{
		int i00 = i & ~pow2q1 & ~pow2q2;
		int i01 = i & ~pow2q1 | pow2q2;
		int i10 = (i | pow2q1) & ~pow2q2;
		int i11 = i | pow2q1 | pow2q2;
		int iq1 = (i & pow2q1) >> shift1;
		int iq2 = (i & pow2q2) >> shift2;
		int iq=(iq1<<1)+iq2;

		out[i] = u[iq][(0<<1)+0] * a[i00] + u[iq][(0<<1)+1] * a[i01] 
                + u[iq][(1<<1)+0] * a[i10] + u[iq][(1<<1)+1] * a[i11];
    }

    return out;
}

complexd* test_adamar_gate(complexd* a, size_t n, size_t k) {
    complexd u[2][2] = {
        complexd(1.0 / pow(2, 0.5)),
        complexd(1.0 / pow(2, 0.5)),

        complexd(1.0 / pow(2, 0.5)),
        complexd(-1.0 / pow(2, 0.5))
    };

    return test_single_q_t(a, n, u, k);
}

complexd* test_n_adamar_gate(complexd* a, size_t n) {
    complexd* out = test_adamar_gate(a, n, 0);
    complexd* temp = out;
    for (size_t i = 1; i < n; i++) {
        out = test_adamar_gate(temp, n, i);
        delete[] temp;
        temp = out;
    }

    return out;
}

complexd* test_rw_gate(complexd* a, size_t n, size_t k, complexd phi) {
    complexd u[2][2] = {
        complexd(1.0),
        complexd(0.0),

        complexd(0.0),
        exp(complexd(0.0, 1.0) * phi)
    };

    return test_single_q_t(a, n, u, k);
}

complexd* test_c_rw_gate(complexd* a, size_t n, size_t k,
                    size_t l, complexd phi) {
    complexd u [4][4] = {
        complexd(1.0), complexd(0.0),
        complexd(0.0), complexd(0.0),

        complexd(0.0), complexd(1.0),
        complexd(0.0), complexd(0.0),

        complexd(0.0), complexd(0.0),
        complexd(1.0), complexd(0.0),

        complexd(0.0), complexd(0.0),
        complexd(0.0), exp(complexd(0.0, 1.0) * phi)
    };

    return test_double_q_t(a, n, u, k, l);
}

complexd* test_not_gate(complexd* a, size_t n, size_t k) {
    complexd u[2][2] = {
        complexd(0.0),
        complexd(1.0),

        complexd(1.0),
        complexd(0.0)
    };

    return test_single_q_t(a, n, u, k);
}

complexd* test_c_not_gate(complexd* a, size_t n, size_t k,
                size_t l) {
    complexd u [4][4] = {
        complexd(1.0), complexd(0.0),
        complexd(0.0), complexd(0.0),

        complexd(0.0), complexd(1.0),
        complexd(0.0), complexd(0.0),

        complexd(0.0), complexd(0.0),
        complexd(0.0), complexd(1.0),

        complexd(0.0), complexd(0.0),
        complexd(1.0), complexd(0.0)
    };

    return test_double_q_t(a, n, u, k, l);
}