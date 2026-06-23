import static header.nn.*;

void main() {
    float[] td = {
        0, 0, 0,
        0, 1, 1,
        1, 0, 1,
        1, 1, 0,
    };

    int n = td.length/3;
    int stride = 3; 
    int offset = 2;
    Mat ti = new Mat(n, 2, stride, 0, td);
    Mat to = new Mat(n, 1, stride, offset, td);
    IO.print(ti);
    IO.print(to);

    int[] arch = {2, 2, 1};
    NN nn = new NN(arch);
    nn_rand(nn, 0, 1);
    IO.print(nn);
}
