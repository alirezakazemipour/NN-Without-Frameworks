#include "optimizers.h"

Optimizer::Optimizer(float lr, vector<Dense*> &params)
{
    this->lr = lr;
    this->parameters = params;
}

void SGD::apply()
{
    for(size_t i = 0; i < this->parameters.size(); i++){
        this->parameters[i]->W = this->utils.mat_add(this->parameters[i]->W,
                                                     this->utils.rescale(this->parameters[i]->dW, -this->lr));
        this->parameters[i]->b = this->utils.mat_add(this->parameters[i]->b,
                                                     this->utils.rescale(this->parameters[i]->db, -this->lr));
    }
}

Momentum::Momentum(float lr, float mu, vector<Dense *> &params) : Optimizer{lr, params}{
    this->mu = mu;
    for(size_t i = 0; i < this->parameters.size(); i++){
        this->gW.push_back(this->utils.rescale(this->parameters[i]->W, 0.0));
        this->gb.push_back(this->utils.rescale(this->parameters[i]->b, 0.0));
    }
}

void Momentum::apply()
{
    for (size_t i = 0; i < this->parameters.size(); i++){
        this->gW[i] = this->utils.mat_add(this->parameters[i]->dW,
                                          this->utils.rescale(this->gW[i], this->mu)
                                          );
        this->parameters[i]->W = this->utils.mat_add(this->parameters[i]->W,
                                                     this->utils.rescale(this->gW[i], -this->lr)
                                                     );
        this->gb[i] = this->utils.mat_add(this->parameters[i]->db,
                                          this->utils.rescale(this->gb[i], this->mu)
                                          );
        this->parameters[i]->b = this->utils.mat_add(this->parameters[i]->b,
                                                     this->utils.rescale(this->gb[i], -this->lr)
                                                     );
    }

}

RMSProp::RMSProp(float lr, float beta, vector<Dense *> &params) : Optimizer(lr, params){
    this->beta = beta;
    for (size_t i = 0; i < this->parameters.size(); i++){
        this->sW.push_back(this->utils.rescale(this->parameters[i]->W, 0.0));
        this->sb.push_back(this->utils.rescale(this->parameters[i]->b, 0.0));

    }

}

void RMSProp::apply()
{
    for (size_t i = 0; i < this->parameters.size(); i++){
        float_batch grad_square_w = this->utils.element_wise_mul(this->parameters[i]->dW, this->parameters[i]->dW);
        grad_square_w = this->utils.rescale(grad_square_w, 1 - this->beta);
        this->sW[i] = this->utils.mat_add(this->utils.rescale(this->sW[i], beta), grad_square_w);
        float_batch grad_step_w = this->utils.element_wise_mul(this->parameters[i]->dW,
                                                               this->utils.element_wise_rev(
                                                                   this->utils.add_scalar(
                                                                       this->utils.element_wise_sqrt(this->sW[i]), eps)
                                                                   )
                                                               );
        this->parameters[i]->W = this->utils.mat_add(this->parameters[i]->W, this->utils.rescale(grad_step_w, -this->lr));

        float_batch grad_square_b = this->utils.element_wise_mul(this->parameters[i]->db, this->parameters[i]->db);
        grad_square_b = this->utils.rescale(grad_square_b, 1 - this->beta);
        this->sb[i] = this->utils.mat_add(this->utils.rescale(this->sb[i], beta), grad_square_b);
        float_batch grad_step_b = this->utils.element_wise_mul(this->parameters[i]->db,
                                                               this->utils.element_wise_rev(
                                                                   this->utils.add_scalar(
                                                                       this->utils.element_wise_sqrt(this->sb[i]), eps)
                                                                   )
                                                               );
        this->parameters[i]->b = this->utils.mat_add(this->parameters[i]->b, this->utils.rescale(grad_step_b, -this->lr));
    }
}

AdaGrad::AdaGrad(float lr, vector<Dense *> &params) : Optimizer(lr, params)
{
    for (size_t i = 0; i < this->parameters.size(); i++){
        this->sW.push_back(this->utils.rescale(this->parameters[i]->W, 0.0));
        this->sb.push_back(this->utils.rescale(this->parameters[i]->b, 0.0));

    }
}

void AdaGrad::apply()
{
    for (size_t i = 0; i < this->parameters.size(); i++){
        float_batch grad_square_w = this->utils.element_wise_mul(this->parameters[i]->dW, this->parameters[i]->dW);
        this->sW[i] = this->utils.mat_add(this->sW[i], grad_square_w);
        float_batch grad_step_w = this->utils.element_wise_mul(this->parameters[i]->dW,
                                                               this->utils.element_wise_rev(
                                                                   this->utils.add_scalar(
                                                                       this->utils.element_wise_sqrt(this->sW[i]), eps)
                                                                   )
                                                               );
        this->parameters[i]->W = this->utils.mat_add(this->parameters[i]->W, this->utils.rescale(grad_step_w, -this->lr));

        float_batch grad_square_b = this->utils.element_wise_mul(this->parameters[i]->db, this->parameters[i]->db);
        this->sb[i] = this->utils.mat_add(this->sb[i], grad_square_b);
        float_batch grad_step_b = this->utils.element_wise_mul(this->parameters[i]->db,
                                                               this->utils.element_wise_rev(
                                                                   this->utils.add_scalar(
                                                                       this->utils.element_wise_sqrt(this->sb[i]), eps)
                                                                   )
                                                               );
        this->parameters[i]->b = this->utils.mat_add(this->parameters[i]->b, this->utils.rescale(grad_step_b, -this->lr));


    }
}
