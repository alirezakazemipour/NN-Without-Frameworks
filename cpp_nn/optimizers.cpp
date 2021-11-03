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
//        cout<<this->parameters[i]->db[0][0]<<endl;


    }
}
