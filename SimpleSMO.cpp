#include<armadillo>
#include<iostream>

double kernel(const arma::Col<double>& x_1, const arma::Col<double>& x_2)
{
    return arma::dot(x_1, x_2);
}

double f_x(const arma::Col<double>& alpha,const arma::Mat<double>& X_1, const arma::Col<double>& x_2, const arma::Col<arma::sword>& Y, const double& b)
{
    double sum {0};

    for(int i =0; i< alpha.n_rows;i++)
    {
        sum = sum + double(alpha(i)*Y(i)*kernel(X_1.col(i),x_2) + b);
    }

    return sum;
}

arma::Col<double> genBoundary(const arma::Col<double>& alpha,const arma::Mat<double>& X_1, const arma::Col<arma::sword>& Y, const double& b)
{

    arma::Col<double> bound(Y.n_rows);

    for(int i = 0; i< Y.n_rows; i++)
    {
        bound(i) = f_x(alpha, X_1, X_1.col(i), Y, b);
    }

    //std::cout << bound;
    return bound;
}

int main()
{

    arma::arma_rng::set_seed_random();

    //Constants
    const int C {100}; // Regularization parameter
    const int max_passes {10};
    const double tol {1};

    double L = 0; // Lower lagrange bound
    double H = 0; // Upper lagrange bound
    double n = 0;
    arma::Mat<double> X_data;
    arma::Col<arma::sword> Y_data;
    arma::Col<double> reclass;

    X_data.load("testData");
    Y_data.load("testClass");

    arma::Col<double> alpha_1(Y_data.n_rows, arma::fill::zeros);
    arma::Col<double> alpha_O(Y_data.n_rows, arma::fill::zeros);
    arma::Col<double> err(Y_data.n_rows, arma::fill::zeros);
    double beta {0};

    int passes {0};

    bool examineAll = 1;
    double numChanged {0};

    while( passes < 1)
    {
        numChanged = 0;

        for(int i = 0; i < Y_data.n_rows; i++)
        {
            //Calculate E_i
            err(i) = f_x(alpha_1, X_data, X_data.col(i), Y_data, beta) - Y_data(i);
            // Check tolerances
            if(((Y_data(i)*err(i) < -tol) && (alpha_1(i)<C)) || ((Y_data(i)*err(i) > tol) && (alpha_1(i) > 0)))
            {
                int j = i;
                // Pick j != i
                while(j == i)
                {
                    //j = rand() % Y_data.n_rows - 1;
                    j = arma::as_scalar(arma::randi<arma::ivec>(1, arma::distr_param(0,Y_data.n_rows-1)));

                }
                // Calculate E_j
                err(j) = f_x(alpha_1, X_data, X_data.col(j), Y_data, beta) - Y_data(j);

                //Store old lagrange multipliers
                alpha_O(i) = alpha_1(i);
                alpha_O(j) = alpha_1(j);

                // Calculate lower and upper bounds for lagrange multipliers

                if(Y_data(i) != Y_data(j))
                {
                    L = std::max(0.0, alpha_1(j) - alpha_1(i));
                    H = std::min(double(C), C + alpha_1(j) - alpha_1(i));
                }
                else{
                    L = std::max(0.0, alpha_1(i) + alpha_1(j) - C);
                    H = std::min(double(C), alpha_1(i) + alpha_1(j));
                }

                if(L != H)
                {

                    n = 2*kernel(X_data.col(i), X_data.col(j)) - kernel(X_data.col(i), X_data.col(i)) - kernel(X_data.col(j), X_data.col(j));
                    if(n<0)
                    {
                        alpha_1(j) = alpha_1(j) - (Y_data(j)*(err(i)-err(j)))/n;

                        // Clip a_j
                        if(alpha_1(j) > H){ alpha_1(j) = H;}
                        else if ( alpha_1(j) < L) { alpha_1(j) = L;}

                        if(std::abs((alpha_1(j)- alpha_O(j))>= 0.000001))
                        {
                            //Compute b-threshold
                            alpha_1(i) = alpha_1(i) + Y_data(i)*Y_data(j)*(alpha_O(j) - alpha_1(j));

                            double beta_1 = beta - err(i) - Y_data(i)*(alpha_1(i) - alpha_O(i))*kernel(X_data.col(i),X_data.col(i)) - Y_data(j)*(alpha_1(j) - alpha_O(j))*kernel(X_data.col(i),X_data.col(j));
                            double beta_2 = beta - err(j) - Y_data(i)*(alpha_1(i) - alpha_O(i))*kernel(X_data.col(i),X_data.col(j)) - Y_data(i)*(alpha_1(j) - alpha_O(j))*kernel(X_data.col(i),X_data.col(j));

                            std::cout << alpha_1(i) << '\t'<< alpha_1(j) << std::endl;

                            if((alpha_1(i) <= 0 || alpha_1(i) >= C) && (alpha_1(j) <= 0 || alpha_1(j) >= C))
                            {
                                beta = (beta_1 + beta_2)/2;

                            }
                            else if(alpha_1(i) > 0 && alpha_1(i) < C)
                            {
                                beta = beta_1;
                                std::cout << beta << std::endl;

                            }
                            else
                            {
                                beta = beta_2;
                            }
                            numChanged = numChanged + 1;

                        }
                    }
                }
            }
        }
        if(numChanged == 0)
        {
            passes = passes +1;
        }
        else
        {
            passes = 0;
        }
    }

    reclass = genBoundary(alpha_1,X_data, Y_data, beta);
    std::cout << reclass;
}
