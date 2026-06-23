import static header.nn.*;

void main() {
    float[] td = {
        0, 0, 0,
        0, 1, 1,
        1, 0, 1,
        1, 1, 0,
    };

    // Random generator = new Random(69);
    // float w1 = generator.nextFloat();
    // float w2 = generator.nextFloat();
    // float b = generator.nextFloat();

    float eps = (float) 1e-1;
    float rate = (float) 1e-1;

    int n = td.length/3;
    int stride = 3; 
    int offset = 2;
    Mat ti = new Mat(n, 2, stride, 0, td);
    Mat to = new Mat(n, 1, stride, offset, td);

    int[] arch = {2, 2, 1};
    NN nn = new NN(arch);
    NN gr = new NN(arch);
    nn_rand(nn, 0, 1);

    System.out.printf("cost = %f\n", nn_cost(nn, ti, to));
    for (int i = 0; i < 20*1000; i++) {
        nn_finite_diff(nn, gr, eps, ti, to);
        nn_learn(nn, gr, rate);
        System.out.printf("cost = %f\n", nn_cost(nn, ti, to));
    }

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            nn.NN_INPUT().MAT_AT(0, 0, i);
            nn.NN_INPUT().MAT_AT(0, 1, j);
            nn_forward(nn);
            System.out.printf("%d ^ %d = %f\n", i, j, nn.NN_OUTPUT().MAT_AT(0, 0));
        }
    }
}

