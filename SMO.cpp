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

//takeStep
int takeStep(int i, int j)
{
/*
takestep
*/


}

//Examine Examples
int examineExample(const arma::Col<double>& alpha, const arma::Mat<double>& X_data, const arma::Col<arma::sword>& Y_data, const arma::Col<double>& cost, double beta, int i)
{
/*
alpha           = lagrange multipliers
X_data          = Training Data
X_sample        = Training sample
Y_data          = Target data
cost            = Cost data
beta            = Threshold
i               = i'th sample
*/
    double tol = 0.0000001;

    double Ei =  Y_data(i)*(f_x(alpha,X_data, X_data.col(i), Y_data, beta) - Y_data(i));

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
                //j = rand() % Y_data.n_rows - 1;
                j = arma::as_scalar(arma::randi<arma::ivec>(1, arma::distr_param(0,Y_data.n_rows-1)));
            }

            if(takeStep(i,j) ==1){
                return 1;
            }
        }

        for(int k = 0; k < alpha.n_rows; k++)
        {

        }




    }
}


//Main routine
int main()
{
    arma::Mat<double> X_data;
    arma::Col<arma::sword> Y_data;
    X_data.load("testData");
    Y_data.load("testClass");

    arma::Col<double> alpha(Y_data.n_rows, arma::fill::zeros);
    arma::Col<double> cost(Y_data.n_rows, arma::fill::zeros);
    cost.fill(100);
    double beta {0};

    int numChanged {0};
    int examineAll {1};

    while( numChanged > 0 | examineAll == 1)
    {
        numChanged = 0;

        if(examineAll == 1)
        {
            for(int i = 0; i< X_data.n_cols; i++)
            {
                numChanged += examineExample(alpha, X_data, Y_data, cost, beta, i);
            }
        }
        else
        {
            for(int i = 0; i< X_data.n_cols; i++)
            {
                if(alpha(i) > 0 && alpha(i) < cost(i))
                numChanged += examineExample(alpha, X_data, Y_data, cost, beta, i);
            }
        }

        if( examineAll == 1) { examineAll = 0; }
        else if (numChanged == 0) { examineAll = 1; }
    }

}
