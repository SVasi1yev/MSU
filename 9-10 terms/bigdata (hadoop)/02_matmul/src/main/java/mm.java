import java.io.*;
import java.util.Iterator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.lib.input.MultipleInputs;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class mm extends Configured implements Tool {
    public static final String MM_TAGS = "ABC";
    public static final String MM_FLOAT_FORMAT = "%.3f";
    public static final String MAPRED_REDUCE_TASKS = "1";
    public static final String MM_GROUPS = "1";

    private static final Logger LOG = Logger.getLogger(mm.class);

    public static int getGroupNum(int i, int n, int grNum) {
        if (grNum >= n) { return i; }
        int d = n / grNum;
        int r = n % grNum;
        if (i >= (d + 1) * r) {
            i -= (d + 1) * r;
            return r + i / d;
        } else {
            return i / (d + 1);
        }
    }

    public static int getGroupSize(int i, int n, int grNum) {
        if (grNum >= n) { return 1; }
        int d = n / grNum;
        int r = n % grNum;
        if (i >= r) {
            return d;
        } else {
            return d + 1;
        }
    }

    public static int getGroupOffset(int i, int n, int grNum) {
        if (grNum >= n) { return i; }
        int d = n / grNum;
        int r = n % grNum;
        if (i >= r) {
            return (d + 1) * r + d * (i - r);
        } else {
            return (d + 1) * i;
        }
    }

    public static void main(String[] args) throws Exception {
        int res = ToolRunner.run(new mm(), args);
        System.exit(res);
    }

    public int run(String[] args) throws Exception {
        Job job = Job.getInstance(getConf(), "mm");
        job.setJarByClass(this.getClass());

        Configuration conf = job.getConfiguration();

        //!!! Наверное лучше объединение путей сделать //TODO
        MultipleInputs.addInputPath(
                job, new Path(args[0] + "/data"),
                TextInputFormat.class, Map.class
        );
        MultipleInputs.addInputPath(
                job, new Path(args[1] + "/data"),
                TextInputFormat.class, Map.class
        );
        FileOutputFormat.setOutputPath(
                job, new Path(args[2] + "/data")
        );

        //        job.setMapperClass(Map.class);
        job.setPartitionerClass(Partition.class);
        job.setSortComparatorClass(KeyComparator.class);
        job.setGroupingComparatorClass(GroupComparator.class);
        job.setReducerClass(Reduce.class);
        job.setNumReduceTasks(Integer.parseInt(
                conf.get("mapred.reduce.tasks", MAPRED_REDUCE_TASKS)
        ));

        job.setMapOutputKeyClass(Key.class);
        job.setMapOutputValueClass(Value.class);
        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(Text.class);
//        job.setOutputFormatClass(TextOutputFormat.class);

        FileSystem fs = FileSystem.get(job.getConfiguration());

        Path aSizeFile = new Path(args[0] + "/size");
        FSDataInputStream is = fs.open(aSizeFile);
        BufferedReader br = new BufferedReader(new InputStreamReader(is));
        String[] splited = br.readLine().split("\t");
        int aRows = Integer.parseInt(splited[0]);
        int aCols = Integer.parseInt(splited[1]);
        conf.setInt("aRows", aRows);
        conf.setInt("aCols", aCols);

        Path bSizeFile = new Path(args[1] + "/size");
        is = fs.open(bSizeFile);
        br = new BufferedReader(new InputStreamReader(is));
        splited = br.readLine().split("\t");
        int bRows = Integer.parseInt(splited[0]);
        int bCols = Integer.parseInt(splited[1]);
        conf.setInt("bRows", bRows);
        conf.setInt("bCols", bCols);

        int cRows = aRows;
        int cCols = bCols;
        Path cSizefile = new Path(args[2] + "/size");
        if (fs.exists(cSizefile)) {
            fs.delete(cSizefile);
        }
        FSDataOutputStream os = fs.create(cSizefile);
        BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(os));
        bw.write(cRows + "\t" + cCols + "\n");
        conf.setInt("cRows", cRows);
        conf.setInt("cCols", cCols);

        bw.close();
        fs.close();

        return job.waitForCompletion(true) ? 0 : 1;
    }

    public static class Key implements WritableComparable<Key> {
        private int groupI;
        private int groupJ;
        private char matName;
        private int i;
        private int j;

        public Key() {}

        public Key(int group_i, int group_j, char matName, int i, int j) {
            this.groupI = group_i;
            this.groupJ = group_j;
            this.matName = matName;
            this.i = i;
            this.j = j;
        }

        @Override
        public void write(DataOutput out) throws IOException {
            out.writeInt(groupI);
            out.writeInt(groupJ);
            out.writeChar(matName);
            out.writeInt(i);
            out.writeInt(j);
        }

        @Override
        public void readFields(DataInput in) throws IOException {
            groupI = in.readInt();
            groupJ = in.readInt();
            matName = in.readChar();
            i = in.readInt();
            j = in.readInt();
        }

        @Override
        public int compareTo(Key o) {
            int res = Integer.compare(groupI, o.groupI);
            if (res != 0) {
                return res;
            } else {
                res = Integer.compare(groupJ, o.groupJ);
                if (res != 0) {
                    return res;
                } else {
                    if ((matName == 'A') & (o.matName == 'B')) {
                        res = Integer.compare(j, o.i);
                        if (res != 0) {
                            return res;
                        } else {
                            return -1;
                        }
                    } else if ((matName == 'B') & (o.matName == 'A')) {
                        res = Integer.compare(i, o.j);
                        if (res != 0) {
                            return res;
                        } else {
                            return 1;
                        }
                    } else if ((matName == 'A') & (o.matName == 'A')) {
                        res = Integer.compare(j, o.j);
                        if (res != 0) {
                            return res;
                        } else {
                            return Integer.compare(i, o.i);
                        }
                    } else {
                        res = Integer.compare(i, o.i);
                        if (res != 0) {
                            return res;
                        } else {
                            return Integer.compare(j, o.j);
                        }
                    }
                }
            }
        }

        @Override
        public int hashCode() {
            return (groupI + "_" + groupJ).hashCode();
        }
    }

    public static class Value implements Writable {
        private char matName;
        private int i;
        private int j;
        private double value;

        public Value() {}

        public Value(char matName, int i, int j, double value) {
            this.matName = matName;
            this.i = i;
            this.j = j;
            this.value = value;
        }

        @Override
        public void write(DataOutput out) throws IOException {
            out.writeChar(matName);
            out.writeInt(i);
            out.writeInt(j);
            out.writeDouble(value);
        }

        @Override
        public void readFields(DataInput in) throws IOException {
            matName = in.readChar();
            i = in.readInt();
            j = in.readInt();
            value = in.readDouble();
        }
    }

    public static class KeyComparator extends WritableComparator {
        protected KeyComparator() {
            super(Key.class, true);
        }

        @Override
        public int compare(WritableComparable wc1, WritableComparable wc2) {
            Key key1 = (Key) wc1;
            Key key2 = (Key) wc2;
            return key1.compareTo(key2);
        }
    }

    public static class GroupComparator extends WritableComparator {
        protected GroupComparator() {
            super(Key.class, true);
        }
        @Override
        public int compare(WritableComparable wc1, WritableComparable wc2) {
            Key key1 = (Key) wc1;
            Key key2 = (Key) wc2;
            int res = Integer.compare(key1.groupI, key2.groupI);
            if (res != 0) {
                return res;
            }
            return Integer.compare(key1.groupJ, key2.groupJ);
        }
    }

    public static class Map extends Mapper<LongWritable, Text, Key, Value> {
        private int groupNum;
        private int aRows;
        private int aCols;
        private int bRows;
        private int bCols;
        private int cRows;
        private int cCols;
        private char aName;
        private char bName;
        private char cName;

        @Override
        protected void setup(Context context) throws IOException {
            Configuration conf = context.getConfiguration();
            groupNum = Integer.parseInt(conf.get("mm.groups", MM_GROUPS));
            aRows = conf.getInt("aRows", -1);
            aCols = conf.getInt("aCols", -1);
            bRows = conf.getInt("bRows", -1);
            bCols = conf.getInt("bCols", -1);
            cRows = conf.getInt("cRows", -1);
            cCols = conf.getInt("cCols", -1);
            String tags = conf.get("mm.tags", MM_TAGS);
            aName = tags.charAt(0);
            bName = tags.charAt(1);
            cName = tags.charAt(2);
        }

        @Override
        public void map(LongWritable offset, Text lineText, Context context)
                throws IOException, InterruptedException {
            String[] split = lineText.toString().split("\t");
            char matName = split[0].charAt(0);
            int i = Integer.parseInt(split[1]);
            int j = Integer.parseInt(split[2]);
            double v = Double.parseDouble(split[3]);

            if (matName == aName) {
                Value value = new Value('A', i, j, v);
                int groupI = getGroupNum(i, aRows, groupNum);
                for (int groupJ = 0; groupJ < groupNum; groupJ++) {
                    Key key = new Key(groupI, groupJ, matName, i, j);
                    context.write(key, value);
                }
            } else {
                Value value = new Value('B', i, j, v);
                int groupJ = getGroupNum(j, bCols, groupNum);
                for (int groupI = 0; groupI < groupNum; groupI++) {
                    Key key = new Key(groupI, groupJ, matName, i, j);
                    context.write(key, value);
                }
            }
        }
    }

    public static class Reduce extends Reducer<Key, Value, NullWritable, Text> {
        int aRows;
        int aCols;
        int bRows;
        int bCols;
        int cRows;
        int cCols;
        int groupNum;
        double[] block;
        double[] aPart;
        double[] bPart;
        char outMatrName;

        NullWritable nw;

        @Override
        public void setup(Context context) throws IOException {
            Configuration conf = context.getConfiguration();
            aRows = conf.getInt("aRows", -1);
            aCols = conf.getInt("aCols", -1);
            bRows = conf.getInt("bRows", -1);
            bCols = conf.getInt("bCols", -1);
            cRows = aRows;
            cCols = bCols;
            groupNum = Integer.parseInt(conf.get("mm.groups", MM_GROUPS));
            outMatrName = conf.get("mm.tags", MM_TAGS).charAt(2);
            nw = NullWritable.get();
        }

        @Override
        public void reduce(Key k, Iterable<Value> values, Context context)
                throws IOException, InterruptedException {
            Configuration conf = context.getConfiguration();
            Iterator<Value> it_values = values.iterator();
//            System.out.println(k.groupI + " --- " + k.groupJ);
//            System.out.println(aCols);
            int nGrSize = getGroupSize(k.groupI, aRows, groupNum);
            int mGrSize = getGroupSize(k.groupJ, bCols, groupNum);
            int nGrOffset = getGroupOffset(k.groupI, cRows, groupNum);
            int mGrOffset = getGroupOffset(k.groupJ, cCols, groupNum);
            block = new double[nGrSize * mGrSize];
            for (int i = 0; i < nGrSize * mGrSize; i++) {
                block[i] = 0;
            }
            aPart = new double[nGrSize];
            bPart = new double[mGrSize];
            Value t = null;
            for (int i = 0; i < aCols; i++) {
                for (int j = 0; j < nGrSize; j++) {
                    if ((t == null) & !(it_values.hasNext())) {
                        aPart[j] = 0.0;
//                        System.out.println("-");
//                        System.out.println("A " + (mGrOffset + j) + " " + i + " " + aPart[j]);
                        continue;
                    }
                    if (t == null) {
                        t = it_values.next();
//                        System.out.println(
//                            t.matName + " " + t.i + " " + t.j + " " + t.value
//                        );
                    }
                    if ((t.matName == 'A') & (t.i == nGrOffset + j) & (i == t.j)) {
                        aPart[j] = t.value;
                        t = null;
//                        System.out.println("+");
                    } else {
                        aPart[j] = 0.0;
//                        System.out.println("-");
                    }
//                    System.out.println("A " + (mGrOffset + j) + " " + i + " " + aPart[j]);
                }
                for (int j = 0; j < mGrSize; j++) {
                    if ((t == null) & !(it_values.hasNext())) {
                        bPart[j] = 0.0;
//                        System.out.println("-");
//                        System.out.println("B " + i + " " + (nGrOffset + j) + " " + bPart[j]);
                        continue;
                    }
                    if (t == null) {
                        t = it_values.next();
//                        System.out.println(
//                                t.matName + " " + t.i + " " + t.j + " " + t.value
//                        );
                    }
                    if ((t.matName == 'B') & (t.j == mGrOffset + j) & (i == t.i)) {
                        bPart[j] = t.value;
                        t = null;
//                        System.out.println("+");
                    } else {
                        bPart[j] = 0.0;
//                        System.out.println("-");
                    }
//                    System.out.println("B " + i + " " + (nGrOffset + j) + " " + bPart[j]);
                }
                for (int n = 0; n < nGrSize; n++) {
                    for (int m = 0; m < mGrSize; m++) {
                        block[n * mGrSize + m] += aPart[n] * bPart[m];
                    }
                }
            }

            for (int n = 0; n < nGrSize; n++) {
                for (int m = 0; m < mGrSize; m++) {
                    if (block[n * mGrSize + m] == 0.0) {
                        continue;
                    }
                    Text value = new Text(
                        outMatrName + "\t" + (nGrOffset + n)
                                + "\t" + (mGrOffset + m)
                                + "\t" + String.format(
                                            conf.get("mm.float-format", MM_FLOAT_FORMAT),
                                            block[n * mGrSize + m]
                                        )
                    );
                    context.write(nw, value);
                }
            }
        }
    }

    public static class Partition extends Partitioner<Key, Value> {
        @Override
        public int getPartition(Key k, Value v, int reducersNum){
            return Math.abs(k.hashCode()) % reducersNum;
        }
    }
}