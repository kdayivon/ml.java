import static header.nn.*;

void main() {
    float[] td = {
        0, 0, 0,
        0, 1, 1,
        1, 0, 1,
        1, 1, 0,
    };

    int n = td.length/3;
    float[] t = {
        0, 0,
        0, 1,
        1, 0,
        1, 1,
    };
    float[] d = {
        0,
        1,
        1,
        0,
    };
    Mat ti = new Mat(n, 2, 2, t);
    IO.print(ti);
    Mat to = new Mat(4, 1, 0, d);
    IO.print(to);

    int[] arch = {2, 2, 1};
    NN nn = new NN(arch);
    nn_rand(nn, 0, 1);
    IO.print(nn);
}
