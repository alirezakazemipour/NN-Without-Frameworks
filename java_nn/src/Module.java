import Layers.Dense;
import java.util.*;
import Losses.Loss;

public abstract class Module {
    public ArrayList<Dense> layers = new ArrayList<>();

    public void backward(Loss loss){
        float[][] delta = loss.delta;
        int num_layers = layers.size();
        for(int i = num_layers - 1; i >= 0; i--){
            delta = this.layers.get(i).backward(delta);
        }

    }
    abstract float[][] forward(float[][] x);
}
