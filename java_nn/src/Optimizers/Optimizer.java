package Optimizers;
import Layers.Dense;

import java.util.ArrayList;

public abstract class Optimizer {
    float lr;
    public ArrayList<Dense> parameters;

    public Optimizer(float lr, ArrayList<Dense> params){
        this.lr = lr;
        this.parameters = params;
    }

    public abstract void apply();
}
