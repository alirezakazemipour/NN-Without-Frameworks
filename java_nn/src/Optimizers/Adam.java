package Optimizers;

import Layers.Dense;
import Utils.Utils;
import java.util.ArrayList;

public class Adam extends Optimizer{
    Utils utils = new Utils();
    float beta1, beta2;
    float eps = (float) Math.pow(10, -8);
    int k = 1;
    ArrayList<float[][]> mW = new ArrayList<>(),
            vW = new ArrayList<>(),
            mb = new ArrayList<>(),
            vb = new ArrayList<>();

    public Adam(ArrayList<Dense> params, float lr, float beta1, float beta2) {
        super(lr, params);
        this.beta1 = beta1;
        this.beta2 = beta2;
        for (Dense param : this.parameters) {
            this.mW.add(utils.rescale(param.W, 0.0F));
            this.vW.add(utils.rescale(param.W, 0.0F));
            this.mb.add(utils.rescale(param.b, 0.0F));
            this.vb.add(utils.rescale(param.b, 0.0F));
        }
    }

    @Override
    public void apply() {
        for (int i = 0; i < this.parameters.size(); i++) {
            this.mW.set(i, utils.mat_add(utils.rescale(this.parameters.get(i).dW,
                    1 - this.beta1), utils.rescale(this.mW.get(i), this.beta1)));
            this.vW.set(i, utils.mat_add(utils.rescale(utils.element_wise_mul(this.parameters.get(i).dW, this.parameters.get(i).dW),
                    1 - this.beta2), utils.rescale(this.vW.get(i), this.beta2)));
            float[][] mW_hat = utils.rescale(this.mW.get(i), 1 / (float)(1 - Math.pow(this.beta1, this.k)));
            float[][] vW_hat = utils.rescale(this.vW.get(i), 1 / (float)(1 - Math.pow(this.beta2, this.k)));
            float[][] grad_step_w = utils.element_wise_mul(mW_hat,
                    utils.element_wise_rev(utils.add_scalar(utils.mat_sqrt(vW_hat), this.eps)));
            this.parameters.get(i).W = utils.mat_add(this.parameters.get(i).W, utils.rescale(grad_step_w, -this.lr));

            this.mb.set(i, utils.mat_add(utils.rescale(this.parameters.get(i).db,
                    1 - this.beta1), utils.rescale(this.mb.get(i), this.beta1)));
            this.vb.set(i, utils.mat_add(utils.rescale(utils.element_wise_mul(this.parameters.get(i).db, this.parameters.get(i).db),
                    1 - this.beta2), utils.rescale(this.vb.get(i), this.beta2)));
            float[][] mb_hat = utils.rescale(this.mb.get(i), (float)(1 - Math.pow(this.beta1, this.k)));
            float[][] vb_hat = utils.rescale(this.vb.get(i), (float)(1 - Math.pow(this.beta2, this.k)));
            float[][] grad_step_b = utils.element_wise_mul(mb_hat,
                    utils.element_wise_rev(utils.add_scalar(utils.mat_sqrt(vb_hat), this.eps)));
            this.parameters.get(i).b = utils.mat_add(this.parameters.get(i).b, utils.rescale(grad_step_b, -this.lr));
        }
        this.k++;
    }
}
