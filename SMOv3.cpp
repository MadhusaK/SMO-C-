#include<armadillo>
#include<iostream>

// Kernel Function
double kernel(const arma::Col<double>& x_1, const arma::Col<double>& x_2)
{
    return arma::dot(x_1, x_2);
}


// SVM Output
double f_x(const arma::Col<double>& alpha,const arma::Mat<double>& X_data, const arma::Col<double>& X_sample, const arma::Col<arma::sword>& Y_data, const double& beta)
{
/*
alpha           = lagrange multipliers
X_data          = Training Data
X_sample        = Training sample
Y_data          = Target data
beta            = Threshold
*/
    double sum {0};

    for(int i =0; i< alpha.n_rows;i++)
    {
        sum += double(alpha(i)*Y_data(i)*kernel(X_data.col(i),X_sample));
    }

    sum -= beta;

    return sum;
}

//takeStep
int takeStep(int i, int j, arma::Col<double>& alpha, const arma::Mat<double>& X_data, const arma::Col<arma::sword>& Y_data, double& beta, const arma::Col<double>& cost, double epsilon, arma::Col<arma::sword>& Y_reclass)
{
/*
i               = First choice
j               = Second Choice
alpha           = lagrange multipliers
X_data          = Training Data
Y_data          = Target data
beta            = Threshold
cost            = Cost data
epislon         = Tolerance for Upper Lower threshold
Y_reclass       = Reclassified data
*/
    arma::arma_rng::set_seed_random();

    // Cover edge cases
    if(i == j) {return 0;}

    // Initialise upper/lower thresholds and errors
    double L {0};
    double H {0};
    double Ei {0};
    double Ej {0};

    double alpha_i {alpha(i)};
    double alpha_j {alpha(j)};

    Ei = (f_x(alpha,X_data, X_data.col(i), Y_data, beta) - Y_data(i));     // Error between the target and actual value
    Ej = (f_x(alpha,X_data, X_data.col(j), Y_data, beta) - Y_data(j));     // Error between the target and actual value
    double s = Y_data(i)*Y_data(j); //



    // Compute Upper nad Lower threshold
    if( Y_data(i) != Y_data(j))
    {
        L = std::max(0.0, alpha(i) - alpha(j));
        H = std::min(cost(i), cost(j) + alpha(i) - alpha(j));
    }
    else
    {
        L = std::max(0.0, alpha(i) + alpha(j) - cost(j));
        H = std::min(cost(i), alpha(i) + alpha(j));
    }

    // Check if the threshold is the same
    if( L == H) { return 0;}

    // Kernel functions
    double kii = kernel(X_data.col(i), X_data.col(i));
    double kjj = kernel(X_data.col(j), X_data.col(j));
    double kij = kernel(X_data.col(i), X_data.col(j));

    double eta =  kii + kjj + 2.0*kij;

    double alpha_i_new;
    double alpha_j_new;

    if(eta > 0)
    {
        alpha_i_new = alpha(i) - Y_data(i)*(Ei - Ej)/eta;

        if( alpha_i_new < L) { alpha_i_new = L;}
        else if (alpha_i_new > H) { alpha_i_new = H;}
    }
    else
    {
        // Compute threshold objetive functions
        double L1 = alpha(j) + s*(alpha(i) - L);
        double H1 = alpha(j) + s*(alpha(i) - H);
        double f1 = Y_data(j)*(Ej + beta) + beta - alpha(j)*kjj - s*alpha(i)*kij;
        double f2 = Y_data(i)*(Ei + beta)+ beta - alpha(i)*kii - s*alpha(j)*kij;

        double L_obj =  -0.5* L1 * L1 * kjj - 0.5* L * L * kii - s * L * L1 * kij - L1 * f1 - L * f2;
        double H_obj =  -0.5* H1 * H1 * kjj - 0.5* H * H * kii - s * H * H1 * kij - H1 * f1 - H * f2;

        if (L_obj > (H_obj + epsilon)) { alpha_i_new = L;}
        else if (L_obj < H_obj - epsilon) { alpha_i_new = H;}
        else { alpha_i_new = alpha(i);}
    }

    // Compare relative error to epsilon
    if (std::abs(alpha_i_new - alpha(i) < epsilon * (alpha(i) * alpha_i_new * epsilon))) { return 0;}

    // Claculate alpha_j_new
    alpha_j_new = alpha(j) + s * (alpha(i) - alpha_i_new);

    if ( alpha_j_new < 0)
    {
        alpha_i_new += s* alpha_j_new;
        alpha_j_new = 0;
    }
    else if ( alpha_j_new > cost(j))
    {
        alpha_i_new += s*(alpha_j_new - cost(j));
        alpha_j_new = cost(j);
    }

    //Approximate percision
    double roundOff = 0.000000001;

    if( alpha_j_new > (cost(j) - roundOff)) { alpha_j_new = cost(j);}
    else if (alpha_j_new < roundOff) { alpha_j_new = 0;}
    if( alpha_i_new > (cost(i) - roundOff)) { alpha_i_new = cost(i);}
    else if (alpha_i_new < roundOff) { alpha_i_new = 0;}


    // Calculate the new threshold
    double b1 {0};
    double b2 {0};
    double b_new {0};

    if( alpha_j_new > 0 && alpha_j_new < cost(j))
    {
        b_new = Ej + Y_data(j)*(alpha_j_new - alpha(j)) * kjj + Y_data(i)*(alpha_i_new - alpha(i)) * kij + beta;
    }
    else
    {
        if (alpha_i_new > 0 && alpha_i_new < cost(i))
        {
        b1 = Ej + Y_data(j)*(alpha_j_new - alpha(j))*kjj + Y_data(j)*(alpha_i_new - alpha(i))*kij + beta;
        b2 = Ei + Y_data(j)*(alpha_j_new - alpha(j))*kij + Y_data(j)*(alpha_i_new - alpha(i))*kjj + beta;
        b_new = (b1 + b2)/2.0;
        }
    }

    beta = b_new;


    alpha(i) = alpha_i_new;
    alpha(j) = alpha_j_new;
    // Update Lagrange

    return 1;
}

//Examine Examples
int examineExample(arma::Col<double>& alpha, const arma::Mat<double>& X_data, const arma::Col<arma::sword>& Y_data, const arma::Col<double>& cost,  double& beta, int i, double epsilon, arma::Col<arma::sword>& Y_reclass)
{
/*
alpha           = lagrange multipliers
X_data          = Training Data
Y_data          = Target data
cost            = Cost data
beta            = Threshold
i               = i'th sample
epislon         = Tolerance for Upper Lower threshold
Y_reclass       = Reclassified data
*/

    double tol = 0.0001;
    int j {-1};

    arma::Col<int> range_vector = arma::linspace<arma::Col<int>>(0, alpha.n_rows-1,alpha.n_rows);

    double Ei = (f_x(alpha,X_data, X_data.col(i), Y_data, beta) - Y_data(i));
    double ri = Y_data(i)*Ei;
    // Heuristic 1
    // KKT Tolerance at the margins
    if(((ri < tol) && (alpha(i) < cost(i))) || ((ri > tol) && (alpha(i) > 1)))
    {

        // Heuristic 2
        // Second choice that maximises the step. This is approximated by finding the maximum error
        double maxErr {0};

        for(int k = 0; k < Y_data.n_rows; k++)
        {

            if((alpha(k) > 0) && (alpha(k) < cost(k)))
            {
                double Ek = (f_x(alpha,X_data, X_data.col(k), Y_data, beta) - Y_data(k));

                if(std::abs(Ek - Ei) > maxErr)
                {
                    maxErr = std::abs(Ek - Ei);
                    j = k;
                }
            }
        }

        if(j >= 0)
        {

            if(takeStep(i,j,alpha, X_data, Y_data, beta, cost, epsilon, Y_reclass) == 1) { return 1;}
        }

        // Heuristic 3
        // Look at non-bounded samples. This covers special cases where Heurisstic 2 fails to make positive progress
        for(int k = 0; k < Y_data.n_rows; k++)
        {
            if((alpha(k) > 0) && (alpha(k) < cost(k)))
            {
                if(takeStep(i,k,alpha, X_data, Y_data, beta, cost, epsilon, Y_reclass) == 1) { return 1;}
            }
        }

        // Heuristic 4
        // Look at alphas at the bounds if heuristic 3 and 4 fails to make positive progress
        for(int k = 0; k < Y_data.n_rows; k++)
        {
            if((alpha(k) <= 0) || (alpha(k) >= cost(k)))
            {
                if(takeStep(i,k,alpha, X_data, Y_data, beta, cost, epsilon, Y_reclass) == 1) { return 1;}
            }
        }
    }

    return 0;


}


//Main routine
int main()
{
    arma::Mat<double> X_data;
    arma::Col<arma::sword> Y_data;

    X_data.load("clusterData");
    Y_data.load("clusterClass");

    std::cout<< "Test";


    arma::Col<arma::sword> Y_reclass(Y_data.n_rows, arma::fill::zeros);
    arma::Col<double> alpha(Y_data.n_rows, arma::fill::zeros);
    arma::Col<double> cost(Y_data.n_rows, arma::fill::zeros);

    cost.fill(100.0);
    double beta {0};
    double epsilon {0.0001};

    int numChanged {0};
    int examineAll {1};

    while( numChanged > 0 || examineAll > 0)
    {
        numChanged = 0;

        if(examineAll > 0)
        {
            for(int i = 0; i< X_data.n_cols; i++)
            {
                numChanged += examineExample(alpha, X_data, Y_data, cost, beta, i, epsilon, Y_reclass);
            }
        }
        else
        {
            for(int i = 0; i< X_data.n_cols; i++)
            {
                if((alpha(i) != 0) && (alpha(i) != cost(i)))
                numChanged += examineExample(alpha, X_data, Y_data, cost, beta, i, epsilon, Y_reclass);
            }
        }

        if( examineAll == 1) { examineAll = 0; }
        else if (numChanged == 0) { examineAll = 1; }
    }

    for(int k =0 ; k < Y_data.n_rows; k++)
    {
        if(f_x(alpha, X_data, X_data.col(k), Y_data, beta) > 1)
        {
            Y_reclass(k) = 1;
        }
        else
        {
            Y_reclass(k) = -1;
        }
    }

    Y_reclass.save("reclassData", arma::csv_ascii);

}

