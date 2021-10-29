package Layers;

interface Layer {
    float[][] forward(float[][] x);
    float[][] backward(float[][] x);
}



