#include <iostream>
#include <module.h>
#include <layers.h>
#include <utils.h>

using namespace std;

class MyNet : Module{
public:
    int in_features = 0;
    Dense *hidden, *output;

    MyNet(int in_features){
        this->in_features = in_features;
        this->hidden = new Dense{this->in_features,
                10,
                "relu",
                "random_uniform",
                "zeros"};
        this->output = new Dense{10,
                1,
                "linear",
                "random_uniform",
                "zeros"};
    }
    vector<vector<float>> forward(const vector<vector<float>> &input){
        vector<vector<float>> x = this->hidden->forward(input);
        x = this->output->forward(x);
        return x;

    }
};

int main()
{
    cout << "Hello World!" << endl;
    vector<vector<float>> x(200, vector<float>(1, 1));


    for(size_t i =0; i < x.size(); i++){
        x[i][0] = 0.01 * i;
    }
    MyNet my_net = MyNet{1};
    vector<vector<float>> y = my_net.forward(x);

    for(size_t i = 0; i < x.size(); i++){
        cout<< y[i][0]<<endl;
    }

    //    Utils utils;
    //    vector<vector<float>> a = {{1, 2}, {3, 4}};
    //    vector<vector<float>> b = {{5, 6}, {7, 8}};
    //    vector<vector<float>> c = utils.mat_mul(a, b);

    //    for(int i = 0; i < 2; i++){
    //        for(int j = 0; j < 2; j++){
    //            cout<< a[i][j]<<endl;

    //        }
    //    }

    return 0;
}
