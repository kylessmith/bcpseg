#ifndef __ONLINE_BCP_H__
#define __ONLINE_BCP_H__

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "src/augmented_interval_array.h"
#include "src/labeled_augmented_array.h"

//-------------------------------------------------------------------------------------

typedef struct {
    double alpha0;
    double *alpha;
    double beta0;
    double *beta;
    double kappa0;
    double *kappa;
    double mu0;
    double *mu;
    int n;
} studentT_t;

//-------------------------------------------------------------------------------------

studentT_t *studentT_init(double alpha, double beta, double kappa, double mu);

double pdf(double x, double df);

double *studentT_pdf(studentT_t *t, double x);

void studentT_update_theta(studentT_t *t, double x);

float *online_changepoint_detection(const double *data, int length, studentT_t *t, double hazard);

aiarray_t *online_bcp_segment(const double *data, int length, double cutoff, double hazard);

void online_bcp_segment_labeled(const double *data, labeled_aiarray_t *segments, char *label, int length, double cutoff, double hazard);

double *online_bcp_probability(const double *data, int length, double cutoff, double hazard, int offset);

aiarray_t *segmentation(double *online_bcp_probability, int length, double cutoff);

void segmentation_labeled(double *online_bcp_probability, labeled_aiarray_t *segments, char *label, int length, double cutoff);

aiarray_t *online_bcp_both(const double *forward_data, const double *reverse_data, int length, double cutoff, double hazard, int offset);

void online_bcp_both_labeled(const double *forward_data, const double *reverse_data, labeled_aiarray_t *segments, char *label, int length, double cutoff, double hazard, int offset);

#endif