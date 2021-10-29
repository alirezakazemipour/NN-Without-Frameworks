package Initializers;

import java.util.Random;

public class RandomUniform implements Initializer {
    public float[][] initialize(int w, int h){
        Random rand = new Random();
        float[][] temp = new float[w][h];
        for(int i = 0; i < w; i++){
            for(int j=0; j < h; j++){
                temp[i][j] = rand.nextFloat();
            }
        }
        return temp;
    }
}
