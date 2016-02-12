#include<armadillo>
#include<iostream>

// Kernel Function
double kernel(const arma::Col<double>& x_1, const arma::Col<double>& x_2)
{
    return arma::dot(x_1, x_2)*arma::dot(x_1, x_2);

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

// Objective function to compute the lower and upper thresholds. The constant is irrelevant as it gets filtered out in the equalities
double w_obj(const double a_1, const double a_2, const double k_11, const double k_22, const double k_12, const double v_1, const double v_2, const int y_1, const int y_2)
{
    return a_1 + a_2 + -0.5*a_1*a_1*k_11 - 0.5*a_2*a_2*k_22 - a_1*a_2*y_1*y_2*k_12 - a_1*y_1*v_1 - a_2*y_2*v_2;
}

int takeStep(int i2, int i1, arma::Col<double>& alpha, const arma::Mat<double>& X_data, const arma::Col<arma::sword>& Y_data, double& beta, const arma::Col<double>& cost, double epsilon)
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

    if(i2 == i1) {return 0;}

    // Initialise constnats to save computational time

    double alpha_i2 {alpha(i2)};
    double alpha_i1 {alpha(i1)};
    int Y_i2 = Y_data(i2);
    int Y_i1 = Y_data(i1);

    // Compute the relative errors
    double E_i2 = double(f_x(alpha,X_data, X_data.col(i2), Y_data, beta) - Y_i2);     // Error between the target and actual value
    double E_i1 = double(f_x(alpha,X_data, X_data.col(i1), Y_data, beta) - Y_i1);     // Error between the target and actual value
    int s = Y_i2*Y_i1;

    /*
    Compute the Upper and Lower thresholds
    */
    double L {0};
    double H {0};

    if( s == 1)
    {
        L = std::max(0.0, alpha_i2 + alpha_i1 - cost(i1));
        H = std::min(cost(i2), alpha_i2 + alpha_i1);
    }
    else
    {
        L = std::max(0.0, alpha_i2 - alpha_i1);
        H = std::min(cost(i2), cost(i1) + alpha_i2 - alpha_i1);
    }

    std::cout << beta << std::endl;

    if(L == H) { return 0;}


    /*
    Compute new Lagrange Multipliers
    */

    double alpha_n_i2 {0};

    double k_11 = kernel(X_data.col(i1), X_data.col(i1));
    double k_22 = kernel(X_data.col(i2), X_data.col(i2));
    double k_12 = kernel(X_data.col(i1), X_data.col(i2));
    double chi = 2.0*k_12 - k_11 - k_22;

    if( chi < 0)
    {
        alpha_n_i2 = alpha_i2 - Y_i2*(E_i1 - E_i2)/chi;

        if( alpha_n_i2 < L) { alpha_n_i2 = L;}
        else if (alpha_n_i2 > H) {alpha_n_i2 = H;}
    }
    else
    {
        double vi1 = f_x(alpha, X_data, X_data.col(i1), Y_data, beta) + beta - alpha_i1*Y_i1*k_11 - alpha_i2*Y_i2*k_12;
        double vi2 = f_x(alpha, X_data, X_data.col(i2), Y_data, beta) + beta - alpha_i1*Y_i1*k_12 - alpha_i2*Y_i2*k_22;
        double L_obj = w_obj(alpha_i1, L, k_11, k_22, k_12, vi1, vi2, Y_i1, Y_i2);
        double H_obj = w_obj(alpha_i1, H, k_11, k_22, k_12, vi1, vi2, Y_i1, Y_i2);

        if( L_obj > (H_obj + epsilon)) { alpha_n_i2 = L;}
        else if ((L_obj < H_obj - epsilon)) { alpha_n_i2 = H;}
        else { alpha_n_i2 = alpha_i2;}
    }

    // Deal with the numerical errors

    if(alpha_n_i2 < epsilon) { alpha_n_i2 = 0.0;}
    else if(alpha_n_i2 > (cost(i2) - epsilon)) { alpha_n_i2 = cost(i2);}
    if(std::abs(alpha_n_i2 - alpha_i2) < epsilon*(alpha_n_i2 + alpha_i2 + epsilon)) { return 0;}

    // Update alpha_i1
    double alpha_n_i1 =  alpha_i1 + s*(alpha_i2 - alpha_n_i2);


    // Update bias
    double beta_n {0.0};

    if((alpha_i1 > 0.0) && (alpha_i1 < cost(i1)))
    {
        beta_n = E_i1 + Y_i1*(alpha_n_i1 - alpha_i1)*k_11 + Y_i2*(alpha_n_i2 - alpha_i2)*k_12 + beta;
    }
    else if((alpha_i2 > 0.0) && ( alpha_i2 < cost(i2)))
    {
        beta_n = E_i2 + Y_i1*(alpha_n_i1 - alpha_i1)*k_12 + Y_i2*(alpha_n_i2 - alpha_i2)*k_22 + beta;
    }
    else
    {
        double b1 = E_i1 + Y_i1*(alpha_n_i1 - alpha_i1)*k_11 + Y_i2*(alpha_n_i2 - alpha_i2)*k_12 + beta;
        double b2 = E_i2 + Y_i1*(alpha_n_i1 - alpha_i1)*k_12 + Y_i2*(alpha_n_i2 - alpha_i2)*k_22 + beta;
        beta_n = (b1 + b2)/2.0;
    }

    // Set bias and new lagrange multipliers
    beta = beta_n;
    alpha(i1) = alpha_n_i1;
    alpha(i2) = alpha_n_i2;

    return 0;
}

//Examine Examples
int examineExample(arma::Col<double>& alpha, const arma::Mat<double>& X_data, const arma::Col<arma::sword>& Y_data, const arma::Col<double>& cost,  double& beta, int i2, double epsilon)
{
/*
alpha           = lagrange multipliers
X_data          = Training Data
X_sample        = Training sample
Y_data          = Target data
cost            = Cost data
beta            = Threshold
i               = i'th sample
epislon         =
*/

    // initialise tolerance and second choice
    double tol = 0.0001;
    int i1 {-1};

    // Initialise target and lagrange to save on computation
    double y_2 = Y_data(i2);
    double alpha_2 = alpha(i2);

    // Compute the relative error
    double E_2 = f_x(alpha, X_data, X_data.col(i2), Y_data, beta) - y_2;
    double r_2 = E_2*y_2;

    //Check KKT conditions
    if( ((r_2 < tol) && (alpha_2 < cost(i2))) || ((r_2 > tol) && (alpha_2 > 0.0)) )
    {
        /*
        Heuristic 2
        1. Check to see if number of non-bounded lagrange multipliers is greater than 1
        2. Choose the second choice by finding the largest relative errori
        */

        // Step 1: Calculate the number of non-bounded lagrange multipliers
        int tot {0};

        for( int k = 0; k < Y_data.n_rows; k++)
        {
            if((alpha(k) > 0.0) && (alpha(k) < cost(k))) { tot++;}
        }

        // Step 2: select j using the second choice heuristic

        if( tot > 1)
        {
            double maxErr {0.0};

            for(int k = 0; k < Y_data.n_rows; k++)
            {

                if((alpha(k) > 0.0) && (alpha(k) < cost(k)))
                {
                    double E_k = (f_x(alpha,X_data, X_data.col(k), Y_data, beta) - Y_data(k));

                    if(std::abs(E_k - E_2) > maxErr)
                    {
                        maxErr = std::abs(E_k - E_2);
                        i1 = k;
                    }
                }
            }

            if(i1 >= 0)
            {
                if(takeStep(i2,i1,alpha, X_data, Y_data, beta, cost, epsilon) == 1) { return 1;}
            }
        }

        /* Heuristic 3
        This is a catch in cases where the second choice heuristic fails to make positive progress
        */

        // Start at a random point
        int rand_point  = arma::as_scalar(arma::randi<arma::ivec>(1, arma::distr_param(0,Y_data.n_rows-1)));

        for(int k = 0; k < Y_data.n_rows; k++)
        {
            // Shift order left using modulo arithmatic in Z_(Y_data.n_rows)
            int k_shift = (k + rand_point) % Y_data.n_rows;

           if( (alpha(k_shift) > 0.0) && (alpha(k) < cost(k_shift)) )
           {
                if(takeStep(i2, k_shift, alpha, X_data, Y_data, beta, cost, epsilon) == 1) { return 1;}
           }
        }


        /* Heuristic 4
        Loop through all largange multipliers if positive progress has yet to be made
        */
        for(int k = 0; k < Y_data.n_rows; k++)
        {
            if((alpha(k) <= 0.0) || (alpha(k) >= cost(k)))
            {
                if(takeStep(i2, k, alpha, X_data, Y_data, beta, cost, epsilon) == 1) { return 1;}
            }
        }

    }

    return 0;

}


int main()
{
    arma::arma_rng::set_seed_random();

    // Load data and respective classifications
    arma::Mat<double> X_data;
    arma::Col<arma::sword> Y_data;

    X_data.load("Data/conData");
    Y_data.load("Data/conClass");
    arma::Col<arma::sword> Y_reclass(Y_data.n_rows, arma::fill::zeros);

    // Initialise Lagrange multipliers and cost
    arma::Col<double> alpha(Y_data.n_rows, arma::fill::zeros);
    arma::Col<double> cost(Y_data.n_rows, arma::fill::zeros);
    cost.fill(100.0);

    // Initialise tolerance and numerical thresholds
    double beta {0.0};
    double epsilon {0.0001};

    // Initialise variables for classification loop
    int numChanged {0};
    bool examineAll {true};

    while( numChanged > 0 || examineAll == true)
    {
        numChanged = 0;

        if(examineAll = true)
        {
            // Loop over every lagrange multiplier if examineAll is true
            for(int i2 = 0; i2< X_data.n_cols; i2++)
            {
                // Increment numChanged if examineExample returns true for i'th multiplier
                numChanged += examineExample(alpha, X_data, Y_data, cost, beta, i2, epsilon);
            }
        }
        else
        {
            // Loop over all non-zero, non-bounded multipliers if examineAll is false
            for(int i2 = 0; i2< X_data.n_cols; i2++)
            {
                if((alpha(i2) > 0) && (alpha(i2) < cost(i2)))
                {
                    numChanged += examineExample(alpha, X_data, Y_data, cost, beta, i2, epsilon);
                }
            }
        }

        if( examineAll == true) { examineAll = false; }
        else if (numChanged == 0) { examineAll = true; }
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

    Y_reclass.save("Data/reclassData", arma::csv_ascii);
}
