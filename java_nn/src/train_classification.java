import Losses.Loss;
import Losses.*;
import Optimizers.*;
import java.util.Random;
import Utils.*;

public class train_classification {
    public static void main(String[] args) {
        Utils utils = new Utils();
        Random random = new Random();
        random.setSeed(1);
        int num_samples = 100;
        int num_features = 2;
        int num_classes = 3;

        int num_epoch = 1000;
        int batch_size = 64;


        float[][] x = new float[num_classes * num_samples][num_features];
        float[][] t = new float[num_classes * num_samples][1];

        float[] radius = new float[num_samples];
        for (int i = 0; i < num_samples; i++) {
            radius[i] = (float) i / num_samples;
        }
        for (int j = 0; j < num_classes; j++) {
            float[] theta = new float[num_samples];
            for (int i = 0; i < num_samples; i++) {
                theta[i] = (float) (i * 0.04 + j * 4 + random.nextGaussian(0, 0.2));
            }
            int k = 0;
            for (int idx = j * num_samples; idx < (j + 1) * num_samples; idx++) {
                x[idx][0] = radius[k] * (float) Math.sin(theta[k]);
                x[idx][1] = radius[k] * (float) Math.cos(theta[k]);
                t[idx][0] = j;
                k++;
            }
        }

        MyNet my_net = new MyNet(num_features, num_classes);
        CrossEntropyLoss mse = new CrossEntropyLoss();
        SGD opt = new SGD(1.0F, my_net.layers);
        float smoothed_loss = 0, total_loss = 0;
        boolean smoothed_flag = false;
        float[][] y;
        for (int epoch = 0; epoch < num_epoch; epoch++) {
            float[][] batch = new float[batch_size][0], target = new float[batch_size][1];
            for (int i = 0; i < batch_size; i++) {
                int idx = random.nextInt(0, x.length);
                batch[i] = x[idx];
                target[i] = t[idx];
            }
            y = my_net.forward(batch);
            Loss loss = mse.apply(y, target);
            my_net.backward(loss);
            opt.apply();
            float reg_loss = 0.0F;
            for(int i = 0; i < my_net.layers.size(); i++){
                float norm2_W = 0;
                int w = my_net.layers.get(i).W.length, h = my_net.layers.get(i).W[0].length;
                for (int k = 0; k < w; k++){
                    for (int l = 0; l < h; l++){
                        norm2_W += Math.pow(my_net.layers.get(i).W[k][l], 2);
                    }
                }
                reg_loss += 0.5 * my_net.layers.get(i).lam * norm2_W;
            }
            total_loss = loss.value + reg_loss;
            if (!smoothed_flag) {
                smoothed_loss = total_loss;
                smoothed_flag = true;
            } else {
                smoothed_loss = (float) (0.9 * smoothed_loss + 0.1 * total_loss);
            }
            System.out.println("Step: " + epoch + " | loss: " + smoothed_loss);
        }
        y = my_net.forward(x);
        int[] predicted_class = new int[batch_size];
        for (int i = 0; i < batch_size; i++) {
            int selected_class = -1;
            float max_prob = -Float.MAX_VALUE;
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
        System.out.println("training acc: " + (float)true_positives / batch_size);
    }
}
