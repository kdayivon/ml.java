package header;

public class NN {
    public static class Mat {
        private int rows;
        private int cols;
        float[] es;

        public Mat(int rows, int cols) {
            this.rows = rows;
            this.cols = cols;
            this.es = new float[rows * cols];
        }

        public Mat(int rows, int cols, float[] es) {
            assert es.length == rows * cols : "array size must match rows * cols";
            this.rows = rows;
            this.cols = cols;
            this.es = es;
        }

        public float MAT_AT(int row, int col) {
            return es[(row) * cols + (col)];
        }        

        public void MAT_AT(int row, int col, float value) {
            es[(row) * cols + (col)] = value;
        }
    }

    public static float rand_float() {
        return (float) Math.random();
    }

    public static void mat_fill(Mat m, float x) {
        for (int i = 0; i < m.rows; i++) {
            for (int j = 0; j < m.cols; j++) {
                m.MAT_AT(i, j, x);
            }
        }
    }
    public static void mat_dot(Mat dst, Mat a, Mat b) {
        if (a.cols != b.rows) throw new IllegalArgumentException("Mats must have same number of columns");
        if (dst.rows != a.rows) throw new IllegalArgumentException("Mats must have same number of columns");
        if (dst.cols != b.cols) throw new IllegalArgumentException("Mats must have same number of columns");
        int n = a.cols;
        for (int i = 0; i < dst.rows; i++) {
            for (int j = 0; j < dst.cols; j++) {
                float x = 0;
                dst.MAT_AT(i, j, x);
                for (int k = 0; k < n; k++) { // iterating over the inner size i.k * k.j
                    x += a.MAT_AT(i, k) * b.MAT_AT(k, j);
                    dst.MAT_AT(i, j, x);
                }
            }
        }
    }
    public static void mat_sum(Mat dst, Mat a) {
        if (a.cols != dst.cols) throw new IllegalArgumentException("Mats must have same number of columns");
        if (a.rows != dst.rows) throw new IllegalArgumentException("Mats must have same number of rows");
        for (int i = 0; i < dst.rows; i++) {
            for (int j = 0; j < dst.cols; j++) {
                float x = dst.MAT_AT(i, j) + a.MAT_AT(i, j);
                dst.MAT_AT(i, j, x);
            }
        }
    }
    public static void mat_rand(Mat m, float low, float high) {
        for (int i = 0; i < m.rows; i++) {
            for (int j = 0; j < m.cols; j++) {
                float x = rand_float() * (high - low) + low;
                m.MAT_AT(i, j, x);
            }
        }
    }

    public static void mat_print(Mat m) {
        for (int i = 0; i < m.rows; i++) {
            for (int j = 0; j < m.cols; j++) {
                System.out.printf("%f ", m.MAT_AT(i, j)); 
            }
            System.out.println();
        }
    } 

}
