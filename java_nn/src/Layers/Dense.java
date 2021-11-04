package Layers;

import Initializers.*;
import Utils.*;
import Activations.*;


public class Dense implements Layer {
    public float[][] W, dW;
    public float[][] b, db;
    int in_features;
    int out_features;
    String weight_initializer = "xavier_uniform";
    String bias_initializer = "zeros";
    Utils utils = new Utils();
    Linear linear = new Linear();
    ReLU relu = new ReLU();
    String act_name = "linear";
    XavierUniform xavier_uniform = new XavierUniform();
    RandomUniform random_uniform = new RandomUniform();
    Constant zeros = new Constant(0.0F);
    float[][] input, z;

    public Dense(int in_features,
                 int out_features,
                 String activation,
                 String weight_initializer,
                 String bias_initializer) {
        this.in_features = in_features;
        this.out_features = out_features;
        this.act_name = activation;
        this.weight_initializer = weight_initializer;
        this.bias_initializer = bias_initializer;

        if (weight_initializer.equals("random_uniform")) {
            this.W = this.random_uniform.initialize(this.in_features, this.out_features);
        }
        else if (weight_initializer.equals("xavier_uniform")) {
            this.W = this.xavier_uniform.initialize(this.in_features, this.out_features);
        }
        if (bias_initializer.equals("zeros")) {
            this.b = this.zeros.initialize(1, this.out_features);
        }

        Constant zero_init = new Constant(0F);
        zero_init.initialize(this.in_features, this.out_features);
        zero_init.initialize(1, this.out_features);


    }

    @Override
    public float[][] forward(float[][] x) {
        this.input = x;
        float[][] z = this.utils.mat_mul(x, this.W);
        float[][] b = new float[z.length][this.b[0].length];
        for (int i = 0; i < z.length; i++) {
            b[i] = this.b[0];
        }
        z = this.utils.mat_add(z, b);
        this.z = z;

        float[][] a = (this.act_name.equals("relu")) ? this.relu.forward(z) : this.linear.forward(z);
        return a;
    }

    @Override
    public float[][] backward(float[][] delta) {
        float[][] dz;
        if (this.act_name.equals("relu")) {
            dz = this.utils.element_wise_mul(delta, this.relu.derivative(this.z));
        }
        else {
            dz = this.utils.element_wise_mul(delta, this.linear.derivative(this.z));
        }

        float[][] input_t = this.utils.transpose(this.input);
        float[][] dw = this.utils.mat_mul(input_t, dz);
        this.dW = this.utils.rescale(dw, 1F / dz.length);

        float[][] ones_t = new float[1][dz.length];
        for(int i = 0; i < ones_t.length; i++){
            for(int j = 0; j < ones_t[0].length; j++){
                ones_t[i][j] = 1;
            }
        }

        float[][] db = utils.mat_mul(ones_t, dz);
        this.db = this.utils.rescale(db, 1F / dz.length);

        float[][] w_t = this.utils.transpose(this.W);
        delta = this.utils.mat_mul(dz, w_t);
        return delta;
    }

    public static void main(String[] args) {
        Utils utils = new Utils();
        float[][] a = new float[][]{{1, 2}, {3, 4}};
        float[][] b = new float[][]{{1, 2}, {3, 4}};

    }
}
