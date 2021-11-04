package Optimizers;

import java.util.ArrayList;
import Layers.Dense;
import Utils.Utils;

public class Momentum extends Optimizer {
    Utils utils = new Utils();
    float mu;
    ArrayList<float[][]> gW = new ArrayList<>(), gb = new ArrayList<>();
    public Momentum(ArrayList<Dense> params, float lr, float mu) {
        super(lr, params);
        this.mu = mu;
        for (Dense param : this.parameters){
            this.gW.add(utils.rescale(param.W, 0.0F));
            this.gb.add(utils.rescale(param.b, 0.0F));

        }
    }

    @Override
    public void apply() {
        for (int i = 0; i < this.parameters.size(); i++) {
            this.gW.set(i, this.utils.mat_add(this.parameters.get(i).dW, utils.rescale(this.gW.get(i), this.mu)));
            this.parameters.get(i).W =
                    this.utils.mat_add(this.parameters.get(i).W, this.utils.rescale(this.gW.get(i), -this.lr));
            this.gb.set(i, this.utils.mat_add(this.parameters.get(i).db, utils.rescale(this.gb.get(i), this.mu)));
            this.parameters.get(i).b =
                    this.utils.mat_add(this.parameters.get(i).b, this.utils.rescale(this.gb.get(i), -this.lr));

        }
    }
}
