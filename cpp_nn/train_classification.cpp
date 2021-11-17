#include <iostream>
#include <module.h>
#include <layers.h>
#include <utils.h>
#include <losses.h>
#include <optimizers.h>

using namespace std;

class MyNet : public Module{
public:
    int in_features = 0, out_features = 0;
    Dense *hidden, *output;

    MyNet(int in_features, int out_features){
        this->in_features = in_features;
        this->out_features = out_features;

        this->hidden = new Dense{this->in_features,
                100,
                "relu",
                "he_normal",
                "zeros",
                "l2",
                0.001};
        this->parameters.push_back(this->hidden);

        this->output = new Dense{100,
                this->out_features,
                "linear",
                "xavier_uniform",
                "zeros",
                "l2",
                0.001};
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
    int num_samples = 500;
    int num_features = 2;
    int num_classes = 3;
    int num_epoch = 1000;
    int batch_size = 64;

    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<> d{0, 0.2};
    std::uniform_int_distribution<> rand_int(0, num_samples * num_classes - 1);

    float_batch x(num_classes * num_samples, vector<float>(num_features, 1));
    float_batch t(num_classes * num_samples, vector<float>(1, 1));

    float radius[num_samples];
    for (int i = 0; i < num_samples; i++){
        radius[i] = i / num_samples;
    }
    for (int j = 0; j < num_classes; j++){
        float theta[num_samples];
        for(int i = 0; i < num_samples; i++){
            theta[i] = i * 0.04 + j * 4 + d(gen);
        }
        int k = 0;
        for (int idx = j * num_samples; idx < (j + 1) * num_samples; idx++){
            x[idx][0] = radius[k] * sin(theta[k]);
            x[idx][1] = radius[k] * cos(theta[k]);
            t[idx][0] = j;
            k++;
        }
    }

    MyNet my_net = MyNet{num_features, num_classes};
    CrossEntropyLoss celoss;
    SGD opt(1, my_net.parameters);
    float smoothed_loss = 0, total_loss = 0;
    bool smoothed_flag = false;
    float_batch y;
    for(int step = 0; step < num_epoch; step++){
        float_batch batch(batch_size, vector<float>(num_features, 1));
        float_batch target(batch_size, vector<float>(1, 1));
        for (int i = 0; i < batch_size; i++) {
            int idx = rand_int(gen);
            batch[i][0] = x[idx][0];
            batch[i][1] = x[idx][1];
            target[i][0] = t[idx][0];
        }


        y= my_net.forward(batch);
        Loss loss = celoss.apply(y, t);
        my_net.backward(loss);
        opt.apply();

        float reg_loss = 0;
        for(size_t i = 0; i < my_net.parameters.size(); i++){
            float norm2_W = 0;
            int w = my_net.parameters[i]->W.size(), h = my_net.parameters[i]->W[0].size();
            for (int k = 0; k < w; k++){
                for (int l = 0; l < h; l++){
                    norm2_W += pow(my_net.parameters[i]->W[k][l], 2);
                }
            }
            reg_loss += 0.5 * my_net.parameters[i]->lambda * norm2_W;
        }

        total_loss = loss.value + reg_loss;
        if (!smoothed_flag) {
            smoothed_loss = total_loss;
            smoothed_flag = true;
        } else {
            smoothed_loss = (float) (0.9 * smoothed_loss + 0.1 * total_loss);
        }

        cout<<"Step: " << step <<" | loss: " << smoothed_loss<<endl;
    }
    y = my_net.forward(x);
    int predicted_class[batch_size];
    for (int i = 0; i < batch_size; i++) {
        int selected_class = -1;
        float max_prob = -std::numeric_limits<float>::max();
        for (int j = 0; j < num_classes; j++) {
            if (y[i][j] > max_prob) {
                max_prob = y[i][j];
                selected_class = j;
            }
        }
        predicted_class[i] = selected_class;
    }
    int true_positives = 0;
    for (int i = 0; i < batch_size; i++) {
        if (predicted_class[i] == (int)t[i][0]) {
            true_positives++;
        }
    }
    cout<<"training acc: " << float(true_positives) / float(batch_size) << endl;
    return 0;

}
