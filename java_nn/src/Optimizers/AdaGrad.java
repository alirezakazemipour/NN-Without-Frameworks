package Optimizers;

import Layers.Dense;
import Utils.Utils;

import java.util.ArrayList;

public class AdaGrad extends Optimizer{
    Utils utils = new Utils();
    float eps = (float) Math.pow(10, -8);
    ArrayList<float[][]> sW = new ArrayList<>(), sb = new ArrayList<>();

    public AdaGrad(ArrayList<Dense> params, float lr) {
        super(lr, params);
        for (Dense param : this.parameters) {
            this.sW.add(utils.rescale(param.W, 0.0F));
            this.sb.add(utils.rescale(param.b, 0.0F));

        }
    }

    @Override
    public void apply() {
        for (int i = 0; i < this.parameters.size(); i++) {
            float[][] grad_square_w = utils.element_wise_mul(this.parameters.get(i).dW, this.parameters.get(i).dW);
            this.sW.set(i, utils.mat_add(this.sW.get(i), grad_square_w));
            float[][] grad_step_w = utils.element_wise_mul(this.parameters.get(i).dW,
                    utils.element_wise_rev(utils.add_scalar(utils.mat_sqrt(this.sW.get(i)), this.eps)
                    )
            );
            this.parameters.get(i).W = utils.mat_add(this.parameters.get(i).W, utils.rescale(grad_step_w, -this.lr));

            float[][] grad_square_b = utils.element_wise_mul(this.parameters.get(i).db, this.parameters.get(i).db);
            this.sb.set(i, utils.mat_add(this.sb.get(i), grad_square_b));
            float[][] grad_step_b = utils.element_wise_mul(this.parameters.get(i).db,
                    utils.element_wise_rev(utils.add_scalar(utils.mat_sqrt(this.sb.get(i)), this.eps)
                    )
            );
            this.parameters.get(i).b = utils.mat_add(this.parameters.get(i).b, utils.rescale(grad_step_b, -this.lr));
        }
    }
}

