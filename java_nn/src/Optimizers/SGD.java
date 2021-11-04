package Optimizers;

import java.util.ArrayList;
import Layers.Dense;
import Utils.Utils;

public class SGD extends Optimizer {
    Utils utils = new Utils();

    public SGD(float lr, ArrayList<Dense> params) {
        super(lr, params);
    }

    @Override
    public void apply() {
        for (Dense parameter : this.parameters) {
            parameter.W = this.utils.mat_add(parameter.W, this.utils.rescale(parameter.dW, -this.lr));
            parameter.b = this.utils.mat_add(parameter.b, this.utils.rescale(parameter.db, -this.lr));

        }
    }
}
