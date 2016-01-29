#include<armadillo>
#include<iostream>

arma::Mat<double> genData(const arma::uword n_dim, const arma::uword n_points, const arma::uword range, const arma::Col<double>& l_slope, const arma::Col<arma::uword>& l_center)
{
/*
n_dim               = Number of dimensions
n_points            = Number of points
l_slope             = Vector perpendicular to desired line
l_center            = Center of the cluster
*/

    const double l_intercept = arma::dot(l_slope, l_center); // intept of the line passing through l_center perpendicular to l_slope

    arma::Mat<double> normData(n_dim, n_points, arma::fill::randn);
    normData = normData*range + range; // Adjust for the new range

    // Variables
    double l_dist {0}; // Perpendicular distance from the line

    for(int i = 0; i< normData.n_cols; i++)
    {
        l_dist = arma::dot(normData.col(i), l_slope) - l_intercept;
        if(abs(l_dist) <= 3)
        {
            normData.shed_col(i);
            i--;
        }

    }
    return normData;
}

arma::Col<arma::sword> tagData(const arma::Mat<double>& data, const arma::Col<double>& l_slope, const arma::Col<arma::uword>& l_center)
{
/*
data                = Number of dimensions
l_slope             = Vector perpendicular to desired line
l_center            = Center of the cluster
*/
    const double l_intercept = arma::dot(l_slope, l_center); // intept of the line passing through l_center perpendicular to l_slope


    double l_dist {0}; // Perpendicular distance from the line
    arma::Col<arma::sword> d_classification(data.n_cols, arma::fill::zeros);


    for(int i = 0; i < data.n_cols; i++)
    {
        l_dist = arma::dot(data.col(i), l_slope) - l_intercept;

        if(l_dist < 0)
        {
            d_classification(i) = -1;
        }
        else
        {
            d_classification(i) = 1;
        }
    }

    return d_classification;
}

int main()
{
    arma::arma_rng::set_seed_random();

    //Constants
    const arma::uword n_dim {2};
    const arma::uword n_points {400};
    const arma::uword range {20};

    const double c_reg_param {2}; // Regularization parameter
    const double tol {2}; // Numerical Tolerance
    const int   max_itr {1}; // Max Iterations

    srand(time(NULL));
    const int theta = std::rand() % 180;
    const arma::Col<double> l_slope = { cos(theta), sin(theta) }; // Vector
    const arma::Col<arma::uword> l_center = {range, range}; // Center of the cluter

    //Variables
    arma::Mat<double> d_testData = genData(n_dim, n_points, range, l_slope, l_center);
    arma::Col<arma::sword> d_classification = tagData(d_testData,l_slope, l_center);

    d_testData.save("testData", arma::csv_ascii);
    d_classification.save("testClass", arma::csv_ascii);

}


