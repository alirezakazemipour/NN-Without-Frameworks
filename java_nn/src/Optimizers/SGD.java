package Optimizers;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Map;

import Layers.Dense;
import Utils.Utils;

public class SGD extends Optimizer {
    Utils utils = new Utils();

    public SGD(float lr, ArrayList<Dense> params) {
        super(lr, params);
    }

    @Override
    public void apply() {
        for (int i = 0; i < this.parameters.size(); i++) {
            this.utils.rescale(this.parameters.get(i).dW, -this.lr);
            this.utils.mat_add(this.parameters.get(i).W, this.parameters.get(i).dW);
            this.utils.rescale(this.parameters.get(i).db, -this.lr);
            this.utils.mat_add(this.parameters.get(i).b, this.parameters.get(i).db);

        }


    }
}
