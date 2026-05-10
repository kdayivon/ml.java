import java.util.Random;

public class Gates {
    final static float[][] or_train = {
        {0, 0, 0},
        {1, 0, 1},
        {0, 1, 1},
        {1, 1, 1}
    };
    final static float[][] and_train = {
        {0, 0, 0},
        {1, 0, 0},
        {0, 1, 0},
        {1, 1, 1}
    };
    final static float[][] nand_train = {
        {0, 0, 1},
        {1, 0, 1},
        {0, 1, 1},
        {1, 1, 0}
    };
    final static float[][] nor_train = {
        {0, 0, 1},
        {1, 0, 0},
        {0, 1, 0},
        {1, 1, 0}
    };
    final static int train_count = 4; 

    // y = x1 * w1 + x2 * w2 + b
    public static float cost(float w1, float w2, float b) {
        float result = 0;
        for (int i = 0; i < train_count; i++) {
            float x1 = or_train[i][0];
            float x2 = or_train[i][1];
            float y = sigmoid((x1*w1) + (x2*w2) + b);
            float diff = y - or_train[i][2];
            result += diff * diff;
        }
        result /= train_count;
        return result;
    }

    /// Compress the value of y between 0 and 1 -> working with binary gates
    public static float sigmoid(float x) {
        float sig = (float)(1 / (1+Math.exp(-x)));
        return sig;
    }

    public static void main(String[] args) {
        Random generator = new Random(69);
        float w1 = generator.nextFloat();
        float w2 = generator.nextFloat();
        float b = generator.nextFloat();

        float eps = (float) 1e-3;
        float rate = (float) 1e-3;
        
        for (int i = 0; i < 1000*1000; i++) {
            float c = cost(w1, w2, b);
            float dw1 = (cost(w1 + eps, w2, b) - c) / eps;
            float dw2 = (cost(w1, w2 + eps, b) - c) / eps;
            float db = (cost(w1, w2, b + eps) - c) / eps;
            w1 -= dw1 * rate;
            w2 -= dw2 * rate;
            b -= db * rate; 
            // System.out.printf("cost = %f, w1 = %f, w2 = %f, b = %f\n", cost(w1,w2,b), w1, w2, b);
        }   
        System.out.printf("cost = %f, w1 = %f, w2 = %f, b = %f\n", cost(w1,w2,b), w1, w2, b);

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                System.out.printf("%d | %d = %.0f\n", i, j, sigmoid(i*w1 + j*w2 + b));
            }
        }
    }
}
