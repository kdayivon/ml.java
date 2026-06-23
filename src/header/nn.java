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
        private int offset;
        float[] es;

        public Mat(int rows, int cols, int stride) {
            this.rows = rows;
            this.cols = cols;
            this.stride = stride;
            this.es = new float[rows * cols];
        }

        public Mat(int rows, int cols, int stride, int offset, float[] es) {
            if (offset + (rows - 1) * stride + cols > es.length) throw new IllegalArgumentException("array size must match rows * cols");
            this.rows = rows;
            this.cols = cols;
            this.stride = stride;
            this.offset = offset;
            this.es = es;
        }
        
        public float MAT_AT(int row, int col) {
            return es[(this.offset) + (row) * stride + (col)];
        }        

        public void MAT_AT(int row, int col, float value) {
            es[(this.offset) + (row) * stride + (col)] = value;
        }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < this.rows; i++) {
                sb.append("     ");
                for (int j = 0; j < this.cols; j++) {
                    sb.append(this.MAT_AT(i, j)).append(" ");
                }
                sb.append("\n");
            }
            return sb.toString(); 
        } 
    }

    public static Mat mat_row(Mat m, int row) {
        return new Mat(1, m.cols, m.stride, m.offset + row * m.stride , m.es);
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
                m.MAT_AT(i, j,x);
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
                for (int k = 0; k < n; k++) { // iterating over the inner size i.k * k.j
                    x += a.MAT_AT(i, k) * b.MAT_AT(k, j);
                }
                dst.MAT_AT(i, j, x);
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
            if( arch.length <= 0) throw new IllegalArgumentException("architecture size must be bigger than 0");
            this.count = (int) arch.length - 1;

           // count = len(architecture) w/o activation layer 
           // allloc ws = 
            this.ws = new Mat[count];
            this.bs = new Mat[count];
            this.as = new Mat[count + 1];
            
            this.as[0] = new Mat(1, arch[0], arch[0]);  // input layer
            for (int i = 1; i < (count+1); i++) {
                this.ws[i-1] = new Mat(arch[i-1], arch[i], arch[i]);
                this.bs[i-1] = new Mat(1, arch[i], arch[i]);
                this.as[i]   = new Mat(1, arch[i], arch[i]);
            }
        }

        public Mat NN_INPUT() {
            return as[0];
        }

        public Mat NN_OUTPUT() {
            return as[count];
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
            mat_dot(nn.as[i+1], nn.as[i], nn.ws[i]); // into the next activation layer, multiply current activation by weight
            mat_sum(nn.as[i+1], nn.bs[i]);           // add the current biases
            mat_sig(nn.as[i+1]);                     // activating the next layer (activation layers is nn.count+1) 
        } 
    }

    public static float nn_cost(NN nn, Mat ti, Mat to) {
        if(ti.rows != to.rows) throw new IllegalArgumentException("training data input and output rows must be the same size");
        if(to.cols != nn.NN_OUTPUT().cols) throw new IllegalArgumentException("expected ouput cols and output cols must be the same size");
        int n = ti.rows;

        float cost = 0;
        for (int i = 0; i < n; i++) {
            Mat x = mat_row(ti, i);     // expected input
            Mat y = mat_row(to, i);     // expected output

            mat_copy(nn.NN_INPUT(), x);
            nn_forward(nn);

            int q = to.cols;
            for (int j = 0; j < q; j++) {
                float d = nn.NN_OUTPUT().MAT_AT(0, j) - y.MAT_AT(0, j);
                cost += d*d;
            }
        }
        return cost/n;
    }

    public static void nn_finite_diff(NN nn, NN gr, float eps, Mat ti, Mat to) {
        float saved;
        float cost = nn_cost(nn, ti, to);
        for (int i = 0; i < nn.count; i++) {
            // weights
            for (int j = 0; j < nn.ws[i].rows; j++) {
                for (int k = 0; k < nn.ws[i].cols; k++) {
                    saved = nn.ws[i].MAT_AT(j, k);
                    float tmp = nn.ws[i].MAT_AT(j, k) + eps; 
                    nn.ws[i].MAT_AT(j, k, tmp);

                    tmp = (nn_cost(nn, ti, to) - cost) / eps;  
                    gr.ws[i].MAT_AT(j, k, tmp);
                    nn.ws[i].MAT_AT(j, k, saved);
                }
            }
            // biases 
            for (int j = 0; j < nn.bs[i].rows; j++) {
                for (int k = 0; k < nn.bs[i].cols; k++) {
                    saved = nn.bs[i].MAT_AT(j, k);
                    float tmp = nn.bs[i].MAT_AT(j, k) + eps; 
                    nn.bs[i].MAT_AT(j, k, tmp);

                    tmp = (nn_cost(nn, ti, to) - cost) / eps;  
                    gr.bs[i].MAT_AT(j, k, tmp);
                    nn.bs[i].MAT_AT(j, k, saved);
                }
            }
        }
    }

    public static void nn_learn(NN nn, NN gr, float rate) {
        for (int i = 0; i < nn.count; i++) {
            // weights
            for (int j = 0; j < nn.ws[i].rows; j++) {
                for (int k = 0; k < nn.ws[i].cols; k++) {
                    float w = nn.ws[i].MAT_AT(j, k);  // learning rate * gradient
                    float g = gr.ws[i].MAT_AT(j, k);  // learning rate * gradient
                    float tmp = w - rate * g;
                    nn.ws[i].MAT_AT(j, k, tmp);
                }
            }
            // biases 
            for (int j = 0; j < nn.bs[i].rows; j++) {
                for (int k = 0; k < nn.bs[i].cols; k++) {
                    float w = nn.bs[i].MAT_AT(j, k); 
                    float g = gr.bs[i].MAT_AT(j, k); 
                    float tmp = w - rate * g;
                    nn.bs[i].MAT_AT(j, k, tmp);
                }
            }
        }
    }

}
