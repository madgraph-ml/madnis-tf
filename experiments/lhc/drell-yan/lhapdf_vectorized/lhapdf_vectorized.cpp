#include "LHAPDF/LHAPDF.h"

extern "C" LHAPDF::PDF* lhapdf_vec_init(char* name, int member) {
    return LHAPDF::mkPDF(name, member);
}

extern "C" void lhapdf_vec_xfxQ2(LHAPDF::PDF* pdf, int id, double* x, double q, double* out,
                                int count) {
    for (int i = 0; i < count; i++) {
        out[i] = pdf->xfxQ2(id, x[i], q);
    }
}

extern "C" double lhapdf_vec_xmax(LHAPDF::PDF* pdf) {
    return pdf->xMax();
}

extern "C" void lhapdf_vec_free(LHAPDF::PDF* pdf) {
    delete pdf;
}
