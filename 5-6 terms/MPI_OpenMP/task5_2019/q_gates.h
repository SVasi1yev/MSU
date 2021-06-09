#pragma once
#include <complex>

using namespace std;
using complexd = complex<double>;

complexd* single_q_t(complexd* a, int n, 
                    complexd u[2][2], int k);

complexd* double_q_t(complexd* a, int n, 
                    complexd u[4][4], int k, int l);

complexd* adamar_gate(complexd* a, int n, int k);

complexd* n_adamar_gate(complexd* a, int n);

complexd* rw_gate(complexd* a, int n, int k, complexd phi);

complexd* c_rw_gate(complexd* a, int n, int k,
                    int l, complexd phi);

complexd* not_gate(complexd* a, int n, int k);

complexd* c_not_gate(complexd* a, int n, int k,
                int l);

complexd* swap_gate(complexd* a, int n, int k,
                int l);