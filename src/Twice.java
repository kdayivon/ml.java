import java.util.Random;

public class Twice {
    final static float[][] train = {
        {0, 0},
        {1, 2},
        {2, 4},
        {3, 6},
        {4, 8}
    };
    final static int train_count = train.length;
    
    // y = x * w + b
    // where w is the weight (value we're trying to find) 
    // where x is the input of the function 
    // where b is the bias (shift function regardless of input value x)
    // where y is the expected value "j" in the pair (i, j)

    public static float cost(float w, float b) {
        float result = 0;
        for (int i = 0; i < train_count; i++) {
            float x = train[i][0];
            float y = x * w + b;
            float diff = y - train[i][1];
            result += diff * diff;
        }
        result /= train_count;
        return result;
    }

    public static void main(String[] args) throws Exception {
        Random generator = new Random(69);
        float w = generator.nextFloat();
        float b = generator.nextFloat();

        float eps = (float) 1e-3; // where eps is the small movement to compute the derivative
        float rate = (float) 1e-3; // where rate is how large of a jump down the curve we make 

        for (int i = 0; i < 5000; i++) {
            float derw = (cost(w + eps, b) - cost(w, b)) / eps; // where der* is the derivative of *
            float derb = (cost(w, b + eps) - cost(w, b)) / eps;
            b -= derb * rate;
            w -= derw * rate;
            System.out.printf("cost = %f, w = %f, b = %f\n", cost(w, b), w, b);
        }
        System.out.printf("Parameter w = %f, Bias b = %f\n", w, b);

        for (int i = 0; i < train_count; i++) {
            float guess = (train[i][0] * w) + b;
            System.out.printf("expected: %f, actual: %.1f\n", train[i][1], guess);
        }
    }
}
