package Utils;

import java.util.Arrays;

public class Utils {
    public float[][] mat_mul(float[][]A, float[][]B){
        int n = A.length, m = A[0].length;
        int k = B.length, l = B[0].length;
//        if(k != m) {
//            throw new RuntimeException("Invalid shape for matrices!");
//        }
        float[][] temp = new float[n][l];
        for(int i = 0; i < n; i++){
            for(int j =0; j < l; j++) {
                temp[i][j] =0;
                for(int r = 0; r < m; r++){
                    temp[i][j] += A[i][r] * B[r][j];
                }
            }
        }
        return temp;
    }

    public void mat_add(float[][]A, float[][]B){
        int n = A.length, m = A[0].length;
        int k = B.length, l = B[0].length;
        if(n != k || m != l) {
            throw new RuntimeException("Invalid shape for matrices!");
        }
        for(int i = 0; i < n; i++){
            for(int j =0; j < l; j++) {
                A[i][j] += B[i][j];
                }
            }
        }

    public static void main(String[] args) {
        float[][] a = new float[][]{{1, 2}, {3, 4}};
        float[][] b = new float[][]{{5, 6}, {7, 8}};
        Utils utils = new Utils();
        System.out.println(Arrays.deepToString(utils.mat_mul(a, b)));
        utils.mat_add(a, b);
        System.out.println(Arrays.deepToString(a));

    }
}
