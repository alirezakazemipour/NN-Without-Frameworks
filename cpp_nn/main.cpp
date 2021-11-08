#include <iostream>
#include <module.h>
#include <layers.h>
#include <utils.h>
#include <losses.h>
#include <optimizers.h>

using namespace std;

class MyNet : public Module{
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
        this->parameters.push_back(this->hidden);
        this->output = new Dense{10,
                1,
                "linear",
                "random_uniform",
                "zeros"};
        this->parameters.push_back(this->output);
    }
    float_batch forward(const float_batch &input){
        float_batch x = this->hidden->forward(input);
        x = this->output->forward(x);
        return x;

    }
};

int main()
{
    cout << "Hello World!" << endl;
    float_batch x(200, vector<float>(1, 1));
    float_batch t(200, vector<float>(1, 1));

    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<> d{0, 0.1};


    for(int i = -100; i < 100; i++){
        //        cout<<i<<endl;
        x[i + 100][0] = 0.01 * i;
        t[i + 100][0] = pow(x[i + 100][0], 2) + d(gen);

    }
    MyNet my_net = MyNet{1};
    MSELoss mse{};
//    Momentum opt(0.2, 0.8, my_net.parameters);
//    SGD opt(0.3, my_net.parameters);
//    RMSProp opt(0.01, 0.99, my_net.parameters);
//    AdaGrad opt(0.05, my_net.parameters);
    Adam opt(0.003, 0.9, 0.999, my_net.parameters);
    float_batch y;

    for(int epoch = 0; epoch < 1000; epoch++){
        y= my_net.forward(x);
        Loss loss = mse.apply(y, t);
        my_net.backward(loss);
        opt.apply();
        //        cout<<my_net.output->b[0][0]<<endl;

        cout<<"Step: " << epoch <<" | loss: " << loss.value<<endl;
    }


    //    for(size_t i = 0; i < x.size(); i++){
    //        cout<<"x = " << x[i][0]<<endl;
    //        cout<<"t = " << t[i][0]<<endl;
    //        cout<<"y = " << y[i][0]<<endl;
    //        cout<< "delta = " << loss.delta[i][0]<<endl;
    //    }

    //    Utils utils;
    //    float_batch a = {{1, 2}, {3, 4}};
    //    float_batch b = {{5, 6}, {7, 8}};
    //    float_batch c = utils.mat_mul(a, b);

    //    for(int i = 0; i < 2; i++){
    //        for(int j = 0; j < 2; j++){
    //            cout<< a[i][j]<<endl;

    //            }
    //        }

    return 0;
}
