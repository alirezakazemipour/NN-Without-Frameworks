package Activations;

interface Activation {
    abstract float[][] forward(float[][] x);
    public float[][] derivative(float[][] x);
}


