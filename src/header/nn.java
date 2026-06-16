package header;

public class nn {
    public static float rand_float() {
        return (float) Math.random();
    }
    
    public static float sigmoidf(float x) {
        float sig = (float)(1 / (1+Math.exp(-x)));
        return sig;
    }

    public static class Mat {
        private int rows;
        private int cols;
        private int stride;
        float[] es;

        public Mat(int rows, int cols, int stride) {
            this.rows = rows;
            this.cols = cols;
            this.stride = stride;
            this.es = new float[rows * cols];
        }

        public Mat(int rows, int cols, int stride, float[] es) {
            assert es.length == rows * cols : "array size must match rows * cols";
            this.rows = rows;
            this.cols = cols;
            this.stride = stride;
            this.es = es;
        }
        
        public float MAT_AT(int row, int col) {
            return es[(row) * stride + (col)];
        }        

        public void MAT_AT(int row, int col, float value) {
            es[(row) * stride + (col)] = value;
        }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < this.rows; i++) {
                sb.append("     ");
                for (int j = 0; j < this.cols; j++) {
                    sb.append(this.MAT_AT(i,j)).append(" ");
                }
                sb.append("\n");
            }
            return sb.toString();
        } 
    }

    public static Mat mat_row(Mat m, int row) {
        float[] l = new float[m.cols];
        for (int i = 0; i < m.cols; i++) {
            l[i] = m.MAT_AT(row, i);
        }
        return new Mat(1, m.cols, m.stride, l);
    }

    public static void mat_copy(Mat dst, Mat src) {
        if (src.cols != dst.cols) throw new IllegalArgumentException("Mats must have same number of columns");
        if (src.rows != dst.rows) throw new IllegalArgumentException("Mats must have same number of rows");
        for (int i = 0; i < dst.rows; i++) {
            for (int j = 0; j < dst.cols; j++) {
                float x = src.MAT_AT(i, j);
                dst.MAT_AT(i, j, x);
            }
        }
    }

    public static void mat_fill(Mat m, float x) {
        for (int i = 0; i < m.rows; i++) {
            for (int j = 0; j < m.cols; j++) {
                m.MAT_AT(i, j, x);
            }
        }
    }

    public static void mat_dot(Mat dst, Mat a, Mat b) {
        if (a.cols != b.rows) throw new IllegalArgumentException("Mats must have same inner sizes");
        if (dst.rows != a.rows) throw new IllegalArgumentException("Mats must have same number of rows");
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

    public static void mat_sig(Mat m) {
        for (int i = 0; i < m.rows; i++) {
            for (int j = 0; j < m.cols; j++) {
                float x = m.MAT_AT(i, j);
                m.MAT_AT(i, j, sigmoidf(x));
            }
        }
    }

    public static class NN {
    int count;    // number of layers
    Mat[] ws;     // array of weights matrices
    Mat[] bs;     // array of bias matrices
    Mat[] as;     // input layer

    // float[] arch = {input_layer = 2, hidden = 2, output = 1}
    // NN nn = new NN(arch, len(arch))
        public NN(int[] arch) {
            assert arch.length > 0: "architecture size must be bigger than 0";
            this.count = (int) arch.length - 1;

           // count = len(architecture) w/o activation layer 
           // allloc ws = 
            this.ws = new Mat[count];
            this.bs = new Mat[count];
            this.as = new Mat[count + 1];
            
            this.as[0] = new Mat(1, arch[0], 0);  // input layer
            for (int i = 1; i < (count+1); i++) {
                this.ws[i-1] = new Mat(arch[i-1], arch[i], 0);
                this.bs[i-1] = new Mat(1, arch[i], 0);
                this.as[i]   = new Mat(1, arch[i], 0);
            }
        }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append("NN = [\n");
            for (int i = 0; i < this.count; i++) {
               sb.append("  ws[").append(i).append("] = [\n").append(ws[i]).append("    ]\n");
               sb.append("  bs[").append(i).append("] = [\n").append(bs[i]).append("    ]\n");
            } 
            sb.append("]\n");
            return sb.toString();
        }
    }
    
    public static void nn_rand(NN nn, float low, float high) {
        for (int i = 0; i < nn.count; i++) {
            mat_rand(nn.ws[i], low, high);
            mat_rand(nn.bs[i], low, high);
        } 
    } 

    public static void nn_forward(NN nn) {
        for (int i = 0; i < nn.count; i++) {
            mat_dot(nn.as[i+1], nn.as[i], nn.ws[i]);
            mat_sum(nn.as[i+1], nn.bs[i]);
            mat_sig(nn.as[i+1]);
        } 
    }
}
