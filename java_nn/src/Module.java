import Layers.Dense;
import java.util.*;

public abstract class Module {
    public ArrayList<Dense> layers = new ArrayList<>();
    public Map<String, float[][]> parameters = new HashMap<>();

    public void set_params(ArrayList<Dense> x){
        int num_layers = x.size();
        for(int i = 0; i < num_layers; i++){
            this.parameters.put("W", x.get(i).W);
            this.parameters.put("b", x.get(i).b);
            this.parameters.put("dW", x.get(i).dW);
            this.parameters.put("db", x.get(i).db);
        }
    }


    abstract float[][] forward(float[][] x);
}
