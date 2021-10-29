import Layers.Dense;
import Losses.Loss;
import Losses.MSELoss;
import Optimizers.SGD;
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
        MyNet my_net = new MyNet(1);
        MSELoss mse = new MSELoss();
        SGD opt = new SGD(0.1F, my_net.layers);
        for (int epoch = 0; epoch < 1000; epoch++){
            float[][] y = my_net.forward(x);
            Loss loss = mse.apply(y, t);
            my_net.backward(loss);
            opt.apply();
            System.out.println("Step: " + epoch +" | loss: " + loss.value);
//            System.out.println(Arrays.deepToString(my_net.hidden1.b));
        }

    }
}
