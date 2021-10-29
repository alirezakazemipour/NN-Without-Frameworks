package Layers;
import Initializers.*;
import Utils.*;
import Activations.*;
import java.util.*;


public class Dense implements Layer {
    public float[][] W, dW;
    public float[][] b, db;
    int in_features;
    int out_features;
    String weight_initializer = "random_uniform";
    String bias_initializer = "zeros";
    Utils utils = new Utils();
    Linear linear = new Linear();
    ReLU relu = new ReLU();
    String act_name = "linear";
    RandomUniform random_uniform = new RandomUniform();
    Constant zeros = new Constant(0F);
    Object input, z;

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

        if(weight_initializer.equals("random_uniform")) {
            this.W = this.random_uniform.initialize(this.in_features, this.out_features);
        }
        if(bias_initializer.equals("zeros")) {
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
        float [][] b = new float[z.length][this.b[0].length];
        for(int i = 0; i < z.length; i++){
            b[i] = this.b[0];
        }
        this.utils.mat_add(z, b);
        this.z = z;

        float[][] a = (this.act_name.equals("relu")) ? this.relu.forward(z) : this.linear.forward(z);
        return a;
    }

    @Override
    public float[][] backward(float[][] x) {
        return new float[0][];
    }

    public static void main(String[] args) {
        Utils utils = new Utils();
        float[][] a = new float[][]{{1, 2}, {3, 4}};
        float[][] b = new float[][]{{1, 2}, {3, 4}};

    }
}
