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
        sum = sum + double(alpha(i)*Y_data(i)*kernel(X_data.col(i),X_sample) + beta);
    }

    return sum;
}

//Classify point




//takeStep
int takeStep(int i, int j, arma::Col<double>& alpha, const arma::Mat<double>& X_data, const arma::Col<arma::sword>& Y_data, double& beta, const arma::Col<double>& cost, double epsilon, arma::Col<arma::sword>& Y_reclass)
{
/*
takestep
*/
    // Possibly uneccessary
    if( i == j) { return 0;}
    double L {0};
    double H {0};
    double Ei {0};
    double Ej {0};
    double alpha_new;

    // Calculate threshold bounds
    int s = Y_data(i)*Y_data(j);
    if(s ==1)
    {
        L = std::max(0.0, alpha(i) + alpha(j) - cost(j));
        H = std::min(cost(i), alpha(i) + alpha(j));
    }
    else
    {
        L = std::max(0.0, alpha(i) - alpha(j));
        H = std::min(cost(i), cost(j) + alpha(i) - alpha(j));
    }

    // Compare Thresholds

    if( L == H) { return 0;}

    double x = kernel(X_data.col(i), X_data.col(i)) + kernel(X_data.col(j), X_data.col(j)) - 2.0*kernel(X_data.col(i), X_data.col(j));

    // Calculate new alpha

    // Calculate the errors in the target
    Ei = (f_x(alpha,X_data,X_data.col(i), Y_data, beta) - Y_data(i));
    Ej = (f_x(alpha,X_data,X_data.col(j), Y_data, beta) - Y_data(j));

    if(x > 0)
    {
        alpha_new = alpha(i) + (1.0/x)*Y_data(i)*( Ej - Ei  );
        alpha_new = std::min(std::max(alpha_new,L), H);
    }
    else if( Y_data(i)*(Ej - Ei) < 0)
    {
        alpha_new = H;
    }
    else
    {
        alpha_new = L;
    }

    if(std::abs(alpha(i) - alpha_new) < epsilon*(epsilon + alpha_new + alpha(i)))
    {
        return 0;
    }

    double alpha_j_old = alpha(j);
    double alpha_i_old = alpha(i);

    // Update Alpha's
    alpha(j) =  alpha(j) + s*(alpha(i) - alpha_new);
    alpha(i) = alpha_new;


    // Recalculate the errors
    Ei = (f_x(alpha,X_data,X_data.col(i), Y_data, beta) - Y_data(i));
    Ej = (f_x(alpha,X_data,X_data.col(j), Y_data, beta) - Y_data(j));

    // Update threshold
    double b1 = beta - Ei - Y_data(i)*(alpha(i) - alpha_i_old)*kernel(X_data.col(i),X_data.col(i)) - Y_data(j)*(alpha(j) - alpha_j_old)*kernel(X_data.col(i), X_data.col(j));
    double b2 = beta - Ej - Y_data(j)*(alpha(i) - alpha_i_old)*kernel(X_data.col(i),X_data.col(j)) - Y_data(j)*(alpha(j) - alpha_j_old)*kernel(X_data.col(j), X_data.col(j));

    beta = (b1+b2)/2.0;
    std::cout << beta;

    // Reclassify data based on new multipliers


}

//Examine Examples
int examineExample(arma::Col<double>& alpha, const arma::Mat<double>& X_data, const arma::Col<arma::sword>& Y_data, const arma::Col<double>& cost,  double& beta, int i, double epsilon, arma::Col<arma::sword>& Y_reclass)
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
    double tol = 0.0000001;
    arma::Col<int> range_vector = arma::linspace<arma::Col<int>>(0, alpha.n_rows-1,alpha.n_rows);

    double Ei =  Y_data(i)*(f_x(alpha,X_data, X_data.col(i), Y_data, beta) - Y_data(i));

    // First heuristic - KKT Conditions w.r.t the margins
    if(( (Ei < -tol) && (alpha(i) < cost(i)) ) || ( (Ei > tol) && (alpha(i) > 0) ))
    {

        // Count number of l.multipliers > 0 and less than C
        int tempSum {0};
        for(int k = 0; k < alpha.n_rows; k++)
        {
            if(alpha(k) > 0 && alpha(k) < cost(k)) {
                tempSum++;
            }
        }
        if( tempSum > 1)
        {
            // Pick j != i
            int j = i;
            while(j == i)
            {
                // Second Choice heuristic
                for(int k = 0; k < alpha.n_rows; k++)
                {
                    if(alpha(k) > 0 && alpha(k) < cost(k))
                    {

                        tempSum++;
                    }
                }
                j = arma::as_scalar(arma::randi<arma::ivec>(1, arma::distr_param(0,Y_data.n_rows-1)));
            }

            if(takeStep(i, j, alpha, X_data, Y_data, beta, cost, epsilon, Y_reclass) ==1){
                return 1;
            }
        }

        // Loop over all possible a_j within the bounds

        for(int j = 0; j < alpha.n_rows; j++)
        {
            // Does it matter if entire order is random instead of looping from a random point??
            arma::Col<int> range_shuff = arma::shuffle(range_vector);
            double k = range_vector(range_shuff(i));

            if (alpha(k)> 0 && alpha(k) < cost(k))
            {
                if(takeStep(i, k, alpha, X_data, Y_data, beta, cost, epsilon, Y_reclass) == 1)
                {
                    return 1;
                }
            }

        }

        // Loop over all possible a_j
        for(int j = 0; j < alpha.n_rows; j++)
        {
            // Does it matter if entire order is random instead of looping from a random point??
            arma::Col<int> range_shuff = arma::shuffle(range_vector);
            double k = range_vector(range_shuff(i));

            if(takeStep(i,k, alpha, X_data, Y_data, beta, cost, epsilon, Y_reclass) == 1)
            {
                return 1;
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
    X_data.load("testData");
    Y_data.load("testClass");

    arma::Col<arma::sword> Y_reclass(Y_data.n_rows, arma::fill::zeros);
    arma::Col<double> alpha(Y_data.n_rows, arma::fill::zeros);
    arma::Col<double> cost(Y_data.n_rows, arma::fill::zeros);
    cost.fill(100);
    double beta {0};
    double epsilon {0.000001};

    int numChanged {0};
    int examineAll {1};

    while( numChanged > 0 | examineAll == 1)
    {
        numChanged = 0;

        if(examineAll == 1)
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
                if(alpha(i) > 0 && alpha(i) < cost(i))
                numChanged += examineExample(alpha, X_data, Y_data, cost, beta, i, epsilon, Y_reclass);
            }
        }

        if( examineAll == 1) { examineAll = 0; }
        else if (numChanged == 0) { examineAll = 1; }
    }

}
