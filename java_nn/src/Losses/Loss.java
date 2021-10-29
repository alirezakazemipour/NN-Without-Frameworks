package Losses;

import java.util.Arrays;

public class Loss {
    float value;
    float[][] delta;
    public Loss(float value, float[][] delta){
        this.value = value;
        this.delta = delta;
    }

    @Override
    public String toString() {
        return "Loss{" +
                "value=" + value +
                ", delta=" + Arrays.deepToString(delta) +
                '}';
    }
}
