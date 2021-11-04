#include "losses.h"

Loss::Loss(float value, float_batch delta)
{
    this->value = value;
    this->delta = delta;
}

Loss MSELoss::apply(float_batch pred, float_batch target)
{
    this->pred = pred;
    this->target = target;
    int w = pred.size(), h = pred[0].size();
    float loss = 0;
    for(int i = 0; i < w; i++){
        for(int j = 0; j < h; j++){
            loss += pow(pred[i][j] - target[i][j], 2) / 2;
        }
    }
    return Loss(loss / w, this->delta());
}

float_batch MSELoss::delta()
{
    float_batch delta = this->utils.mat_add(this->pred, this->utils.rescale(this->target, -1));
    return delta;
}
