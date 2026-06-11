import static header.NNbasic.*;

void main() {
    Mat a = new Mat(1, 2);
    mat_rand(a, 5, 10);
    mat_print(a);
    float[] id_data = {1, 0, 0, 1};

    Mat b = new Mat(2, 2, id_data);

    System.out.println("-----------------");
    IO.println(b.MAT_AT(1, 1));
    Mat dst = new Mat(1, 2);
    mat_dot(dst, a, b);
    mat_print(dst);

}
