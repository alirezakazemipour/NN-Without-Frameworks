import Layers.Dense;
import Losses.Loss;
import Losses.MSELoss;
import Optimizers.*;
import java.util.Random;
import java.lang.Math;

class MyNet extends Module{
    int in_features = 0, out_features = 0;
    Dense hidden1, output;
    public MyNet(int in_features, int out_features){
        this.in_features = in_features;
        this.out_features = out_features;
        this.hidden1 = new Dense(this.in_features,
                100,
                "relu",
                "he_normal",
                "zeros",
                "l2",
                0.001F);
        this.layers.add(this.hidden1);
        this.output = new Dense(100,
                out_features,
                "linear",
                "xavier_uniform",
                "zeros",
                "l2",
                0.001F);
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
        int num_epoch = 1000;
        int batch_size = 200;

        float[][] x = new float[200][1], t = new float[200][1];
        for(int i = -100; i < 100; i++){
            x[i + 100][0] = 0.01F * i;
            t[i + 100][0] = (float) Math.pow(x[i + 100][0], 2) + (float)(random.nextGaussian() * 0.1);
        }
        MyNet my_net = new MyNet(1, 1);
        MSELoss mse = new MSELoss();
        Adam opt = new Adam(my_net.layers, 0.001F, 0.9F, 0.999F);
        float smoothed_loss =0;
        boolean smoothed_flag = false;
        for (int epoch = 0; epoch < num_epoch; epoch++){
            float[][] batch = new float[batch_size][0], target = new float[batch_size][1];
            for (int i = 0; i <batch_size; i++){
                int idx = random.nextInt(0, x.length);
                batch[i] = x[idx];
                target[i] = t[idx];
            }
            float[][] y = my_net.forward(batch);
            Loss loss = mse.apply(y, target);
            my_net.backward(loss);
            opt.apply();
            if(!smoothed_flag){
                smoothed_loss = loss.value;
                smoothed_flag = true;
            }
            else {
                smoothed_loss = (float)(0.9 * smoothed_loss + 0.1 * loss.value);
            }
            System.out.println("Step: " + epoch +" | loss: " + smoothed_loss);
        }
    }
}
