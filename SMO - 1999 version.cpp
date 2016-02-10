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
}


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
