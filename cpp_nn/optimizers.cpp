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
