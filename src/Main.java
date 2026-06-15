import static header.nn.*;

void main() {
    float[] td = {
        0, 0, 0,
        0, 1, 1,
        1, 0, 1,
        1, 1, 0,
    };


    int[] arch = {2, 2, 1};
    NN nn = new NN(arch);
    nn_rand(nn, 0, 1);
    IO.print(nn);
}
