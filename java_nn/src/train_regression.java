import Layers.Dense;
import Losses.Loss;
import Losses.MSELoss;
import java.util.Arrays;
import java.util.Random;
import java.lang.Math;

class MyNet extends Module{
    int in_features = 0;
    Dense hidden1, output;
    public MyNet(int in_features){
        this.in_features = in_features;
        this.hidden1 = new Dense(this.in_features,
                10,
                "relu",
                "random_uniform",
                "zeros");
        this.layers.add(this.hidden1);
        this.output = new Dense(10,
                1,
                "linear",
                "random_uniform",
                "zeros");
        this.layers.add(this.output);
        this.set_params(this.layers);
    }

    public float[][] forward(float[][] x){
        x = this.hidden1.forward(x);
        x = this.output.forward(x);
        return x;
    }
}

public class train_regression {
    public static void main(String[] args) {
        Random random = new Random();
        random.setSeed(1);

        float[][] x = new float[200][1], t = new float[200][1];
        for(int i = -100; i < 100; i++){
            x[i + 100][0] = 0.01F * i;
            t[i + 100][0] = (float) Math.pow(x[i + 100][0], 2) + (float)(random.nextGaussian() * 0.1);
        }
        System.out.println(Arrays.deepToString(x));
        MyNet my_net = new MyNet(1);
        float[][] y = my_net.forward(x);
        System.out.println(Arrays.deepToString(y));
        System.out.println(Arrays.deepToString(t));
        MSELoss mse = new MSELoss();
        Loss loss = mse.apply(y, t);
        System.out.println(loss);




    }
}
