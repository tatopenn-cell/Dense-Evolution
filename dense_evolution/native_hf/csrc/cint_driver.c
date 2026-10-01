#include <stdlib.h>
#include "cint.h"
#include "cint_funcs.h"

#ifdef _WIN32
#define DE_EXPORT __declspec(dllexport)
#else
#define DE_EXPORT
#endif

static int ncart(int *bas, int ib)
{
    int l = bas[ib * BAS_SLOTS + ANG_OF];
    return (l + 1) * (l + 2) / 2;
}

DE_EXPORT void de_int1e(int kind, double *out, int n, int *ao_loc, double *r, int *pos,
                        int *atm, int natm, int *bas, int nbas, double *env)
{
    double buf[36];
    int shls[2];
    for (int i = 0; i < nbas; i++) {
        for (int j = 0; j < nbas; j++) {
            int di = ncart(bas, i), dj = ncart(bas, j);
            shls[0] = i;
            shls[1] = j;
            if (kind == 0) int1e_ovlp_cart(buf, NULL, shls, atm, natm, bas, nbas, env, NULL, NULL);
            else if (kind == 1) int1e_kin_cart(buf, NULL, shls, atm, natm, bas, nbas, env, NULL, NULL);
            else int1e_nuc_cart(buf, NULL, shls, atm, natm, bas, nbas, env, NULL, NULL);
            for (int b = 0; b < dj; b++) {
                for (int a = 0; a < di; a++) {
                    int A = ao_loc[i] + a, B = ao_loc[j] + b;
                    out[(size_t)pos[A] * n + pos[B]] = buf[a + di * b] * r[A] * r[B];
                }
            }
        }
    }
}

DE_EXPORT void de_int2e(double *out, int n, int *ao_loc, double *r, int *pos,
                        int *atm, int natm, int *bas, int nbas, double *env)
{
    CINTOpt *opt = NULL;
    double buf[1296];
    int shls[4];
    size_t N = n;
    int2e_optimizer(&opt, atm, natm, bas, nbas, env);
    for (int i = 0; i < nbas; i++) {
        for (int j = 0; j <= i; j++) {
            for (int k = 0; k <= i; k++) {
                int lmax = k < i ? k : j;
                for (int l = 0; l <= lmax; l++) {
                    int di = ncart(bas, i), dj = ncart(bas, j), dk = ncart(bas, k), dl = ncart(bas, l);
                    shls[0] = i;
                    shls[1] = j;
                    shls[2] = k;
                    shls[3] = l;
                    int2e_cart(buf, NULL, shls, atm, natm, bas, nbas, env, opt, NULL);
                    for (int d = 0; d < dl; d++) {
                        for (int c = 0; c < dk; c++) {
                            for (int b = 0; b < dj; b++) {
                                for (int a = 0; a < di; a++) {
                                    int A = ao_loc[i] + a, B = ao_loc[j] + b, C = ao_loc[k] + c, D = ao_loc[l] + d;
                                    double v = buf[a + di * (b + dj * (c + dk * d))] * r[A] * r[B] * r[C] * r[D];
                                    size_t p = pos[A], q = pos[B], s = pos[C], t = pos[D];
                                    out[((p * N + q) * N + s) * N + t] = v;
                                    out[((q * N + p) * N + s) * N + t] = v;
                                    out[((p * N + q) * N + t) * N + s] = v;
                                    out[((q * N + p) * N + t) * N + s] = v;
                                    out[((s * N + t) * N + p) * N + q] = v;
                                    out[((t * N + s) * N + p) * N + q] = v;
                                    out[((s * N + t) * N + q) * N + p] = v;
                                    out[((t * N + s) * N + q) * N + p] = v;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    CINTdel_optimizer(&opt);
}
