#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

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

Mat RL_deconvolution(Mat observed, Mat psf, int iterations) {

	// Uniform grey starting estimation
	Mat latent_est = Mat(observed.size(), CV_64FC1, 0.5);

	// Flip the point spread function (NOT the inverse)
	Mat psf_hat = Mat(psf.size(), CV_64FC1);
	int psf_row_max = psf.rows - 1;
	int psf_col_max = psf.cols - 1;
	for (int row = 0; row <= psf_row_max; row++) {
		for (int col = 0; col <= psf_col_max; col++) {
			psf_hat.at<double>(psf_row_max - row, psf_col_max - col) =
				psf.at<double>(row, col);
		}
	}

	Mat est_conv;
	Mat relative_blur;
	Mat error_est;

	// Iterate
	for (int i=0; i<iterations; i++) {

		filter2D(latent_est, est_conv, -1, psf);

		// Element-wise division
		relative_blur = observed.mul(1.0/est_conv);

		filter2D(relative_blur, error_est, -1, psf_hat);

		// Element-wise multiplication
		latent_est = latent_est.mul(error_est);
	}

	return latent_est;
}

int main( int argc, const char** argv )
{

	if (argc != 3) {
		cout << "Usage: " << argv[0] << " image iterations" << "\n";
		return -1;
	}

	int iterations = atoi(argv[2]);

	// Read the original image
	Mat original_image;
	original_image = imread(argv[1], CV_LOAD_IMAGE_UNCHANGED);

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
			return -2;
	}

	// From here on, use 64-bit floats
	// Convert original_image to float
	Mat float_image;
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
	Mat psf = Mat(Size(psf_size, psf_size), CV_64FC1, 0.0);

	for (int j = 0; j<psf.rows; j++) {
		for (int k = 0; k<psf.cols; k++) {
			temp = exp(
					-0.5 * (
						pow((j - mean_row) / sigma_row, 2.0) + 
						pow((k - mean_col) / sigma_col, 2.0))) /
				(2* M_PI * sigma_row * sigma_col);
			sum += temp;
			psf.at<double>(j,k) = temp;
		}
	}

	// Normalise the psf.
	for (int row = 0; row<psf.rows; row++) {
		// cout << row << " ";
		for (int col = 0; col<psf.cols; col++) {
			psf.at<double>(row, col) /= sum;
			// cout << psf.at<double>(row, col) << " ";
		}
		// cout << "\n";
	}

	// Blur the float_image with the psf.
	Mat blurred_float;
	blurred_float = float_image.clone();
	filter2D(float_image, blurred_float, -1, psf);
	namedWindow("BlurredFloat", CV_WINDOW_AUTOSIZE);
	imshow("BlurredFloat", blurred_float);

	Mat estimation = RL_deconvolution(blurred_float, psf, iterations);
	namedWindow("Estimation", CV_WINDOW_AUTOSIZE);
	imshow("Estimation", estimation);

	waitKey(0); //wait infinite time for a keypress

	destroyWindow("Float");
	destroyWindow("BlurredFloat");
	destroyWindow("Estimation");

	return 0;
}