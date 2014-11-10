#include "math.h"
#include <opencv/cxcore.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <stdio.h>

// From wikipedia:
//
// def RL_deconvolution(observed, psf, iterations):
//     # initial estimate is arbitrary - uniform 50% grey works fine
//     latent_est = 0.5*np.ones(observed.shape)
//     # create an inverse psf
//     psf_hat = psf[::-1,::-1]
//     # iterate towards ML estimate for the latent image
//     for i in np.arange(iterations):
//         est_conv      = cv2.filter2D(latent_est,-1,psf)
//         relative_blur = observed/est_conv;
//         error_est     = cv2.filter2D(relative_blur,-1,psf_hat)
//         latent_est    = latent_est * error_est
//     return latent_est

static int image_type;

CvMat* RL_deconvolution(CvMat* observed, CvMat* psf, int iterations) {

	CvScalar grey;
	int row;
	int col;
	int i;

	// Uniform grey starting estimation
	switch (image_type) {
		case CV_64FC1:
			grey.val[0] = 0.5;
		case CV_64FC3:
			grey.val[0] = 0.5;
			grey.val[1] = 0.5;
			grey.val[2] = 0.5;
	}
	CvMat* latent_est = cvCreateMat(observed->rows, observed->cols, CV_64FC1);
	cvSet(latent_est, grey, NULL);

	// Flip the point spread function (NOT the inverse)
	CvMat* psf_hat = cvCreateMat(psf->rows, psf->cols, CV_64FC1);
	int psf_row_max = psf->rows - 1;
	int psf_col_max = psf->cols - 1;
	for (row = 0; row <= psf_row_max; row++) {
		for (col = 0; col <= psf_col_max; col++) {
			cvSet2D(psf_hat, psf_row_max - row, psf_col_max - col, cvGet2D(psf, row, col));
		}
	}

	CvMat* est_conv;
	CvMat* relative_blur;
	CvMat* error_est;

	// Iterate
	for (i=0; i<iterations; i++) {

		// filter2D(latent_est, est_conv, -1, psf);
		cvFilter2D(latent_est, est_conv, psf, cvPoint(-1,-1));

		// Element-wise division
		// relative_blur = observed.mul(1.0/est_conv);
		cvDiv(observed, est_conv, relative_blur, 1.0);

		// filter2D(relative_blur, error_est, -1, psf_hat);
		cvFilter2D(relative_blur, error_est, psf_hat, cvPoint(-1,-1));

		// Element-wise multiplication
		// latent_est = latent_est.mul(error_est);
		cvMul(latent_est, error_est, latent_est, 1.0);
	}

	return latent_est;
}

int main( int argc, const char** argv )
{

	int row;
	int col;

	if (argc != 3) {
		printf("Usage: %s image iterations\n", argv[0]);
		return -1;
	}

	int iterations = atoi(argv[2]);

	// Read the original image
	CvMat* original_image;
	original_image = cvLoadImageM(argv[1], CV_LOAD_IMAGE_UNCHANGED);

	int num_channels = CV_MAT_CN(original_image->type);
	switch (num_channels) {
		case 1:
			image_type = CV_64FC1;
			break;
		case 3:
			image_type = CV_64FC3;
			break;
		default:
			return -2;
	}

	// This is a hack, assumes too much
	int divisor;
	switch (original_image.elemSize()) {
		case 1:
			divisor = 255;
			break;
		case 2:
			divisor = 65535;
			break;
		default:
			return -3;
	}

	// From here on, use 64-bit floats
	// Convert original_image to float
	CvMat* float_image;
	original_image.convertTo(float_image, CV_64FC1);
	float_image *= 1.0/divisor;
	namedWindow("Float", CV_WINDOW_AUTOSIZE);
	imshow("Float", float_image);

	// Calculate a gaussian blur psf.
	double sigma_row = 9.0;
	double sigma_col = 5.0;
	int psf_size = 5;
	double mean_row = 0.0;
	double mean_col = psf_size/2.0;
	double sum = 0.0;
	double temp;
	CvMat* psf = cvCreateMat(psf_size, psf_size, CV_64FC1);
	CvScalar zeros;
	zeros.val[0] = 0.0;
	cvSet(psf, zeros, NULL);

	for (row = 0; row < psf->rows; row++) {
		for (col = 0; col < psf->cols; col++) {
			temp = exp(
					-0.5 * (
						pow((row - mean_row) / sigma_row, 2.0) + 
						pow((col - mean_col) / sigma_col, 2.0))) /
				(2* M_PI * sigma_row * sigma_col);
			sum += temp;
			cvSetReal2D(psf, row, col, temp);
		}
	}

	// Normalise the psf.
	for (row = 0; row < psf->rows; row++) {
		for (col = 0; col < psf->cols; col++) {
			cvSetReal2D(psf, row, col, cvGetReal2D(psf, row, col) / sum);
		}
	}

	// Blur the float_image with the psf.
	CvMat* blurred_float;
	blurred_float = cvCloneMat(float_image);
	// filter2D(float_image, blurred_float, -1, psf);
	cvFilter2D(float_image, blurred_float, psf, cvPoint(-1,-1));
	namedWindow("BlurredFloat", CV_WINDOW_AUTOSIZE);
	imshow("BlurredFloat", blurred_float);

	CvMat* estimation = RL_deconvolution(blurred_float, psf, iterations);
	namedWindow("Estimation", CV_WINDOW_AUTOSIZE);
	imshow("Estimation", estimation);

	waitKey(0); //wait infinite time for a keypress

	destroyWindow("Float");
	destroyWindow("BlurredFloat");
	destroyWindow("Estimation");

	return 0;
}
