#include<iostream>
#include<armadillo>
#include<math.h>

arma::Mat<double> genConData(const arma::uword n_dim, const arma::uword n_points, const arma::uword range)
{
/*
n_dim               = Number of dimensions
n_points            = Number of points
range               = Range of the dataset
d_POS_classifier    = Symbol for the positive classifier
d_NEG_classifier    = Symbol for the negative classifier
*/
    arma::Mat<double> conCirc(n_dim, n_points, arma::fill::randu); // Matrix containing datapoints
    conCirc = conCirc*range - range/2.0; //Change the range of the dataset and center at 0

    arma::uword points = n_points;


    for(int i = 0; i < conCirc.n_cols; i++)
    {
        if((arma::norm(conCirc.col(i)) > range*(1.0/6.0)) && (arma::norm(conCirc.col(i)) < range*(2.0/6.0)))
        {
            conCirc.shed_col(i); // Remove coloumn of it falls within the separation circle
            i = i -1;
        }
    }

    return conCirc;
}

arma::Col<arma::sword> classConData(const arma::Mat<double>& data, const arma::uword range)
{
    arma::Col<arma::sword> classifierType(data.n_cols, arma::fill::zeros); // Vector containing groundtruth callisifiers
    // Ground truth classification
    for(int i = 0; i < data.n_cols; i++)
    {
        if(arma::norm(data.col(i)) < range*(2.0/6.0))
        {
            classifierType(i) = 1; // Positive Classifier
        }
        else
        {
            classifierType(i) = -1; // Negative Classifier
        }
    }

    return classifierType;
}



///Generate Clusters
void genCluster(const arma::uword nPoints, const arma::uword nDim, const arma::uword separation)
{
/*
nPoints             = Number of points in each cluster
nDim                = The number of dimenions of each point
nCluster            = Number of clusters
separation          = The separation of ecah cluster
*/
    arma::Mat<double> dataPoints;
    arma::Col<arma::sword> classification;

    for(int i =0; i < 2; i++)
    {

        arma::Mat<double> tempPoints(nDim, nPoints, arma::fill::randn);
        arma::Col<arma::sword> cluster(nPoints);
        cluster.fill(-1 + i*2);

        //cluster.fill(i);
        tempPoints += i*(separation/2)/2;

        classification = arma::join_vert(classification, cluster);
        //tempPoints = arma::join_vert(tempPoints,cluster);
        dataPoints = arma::join_horiz(dataPoints, tempPoints);
    }

    dataPoints.save("clusterData", arma::csv_ascii);
    classification.save("clusterClass", arma::csv_ascii);
    //return dataPoints;
}

int main()
{
    arma::arma_rng::set_seed_random();

    //Constants
    const arma::uword n_dim {2};
    const arma::uword n_points {400};
    const arma::uword range {20};


    //Variables
    arma::Mat<double> d_2D_conData = genConData(n_dim,n_points, range); // Create nonlinearly separable data
    arma::Mat<arma::sword> d_2D_conClass = classConData(d_2D_conData,range); // Create nonlinearly separable data

    genCluster(200, 2, 15);

    d_2D_conData.save("conData", arma::csv_ascii);
    d_2D_conClass.save("conClass", arma::csv_ascii);

}
