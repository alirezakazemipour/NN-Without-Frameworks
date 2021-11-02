#include <iostream>
#include <initializers.h>

using namespace std;

int main()
{
    cout << "Hello World!" << endl;
    Constant cons(3);
    RandomUniform ru;
    float** arr = cons.initialize(2, 2);
    float** arr2 = ru.initialize(2, 2);
    for(int i = 0; i < 2; i++){
        for(int j = 0; j < 2; j++){
            cout<<"pointer " << *arr2[i]<<endl;
            cout<<"normal " << arr2[i][j]<<endl;
            cout<<"----"<<endl;

        }
    }
    return 0;
}
