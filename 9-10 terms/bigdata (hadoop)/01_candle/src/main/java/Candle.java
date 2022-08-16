import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.math.BigDecimal;
import java.util.*;
import java.util.regex.Pattern;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class Candle extends Configured implements Tool {
    public static final String CANDLE_WIDTH = "300000";
    public static final String CANDLE_SECURITIES = ".*";
    public static final String CANDLE_DATE_FROM = "19000101";
    public static final String CANDLE_DATE_TO = "20200101";
    public static final String CANDLE_TIME_FROM = "1000";
    public static final String CANDLE_TIME_TO = "1800";
    public static final String CANDLE_NUM_REDUCER = "1";

    private static final Logger LOG = Logger.getLogger(Candle.class);

    public static void main(String[] args) throws Exception {
        int res = ToolRunner.run(new Candle(), args);
        System.exit(res);
    }

    public int run(String[] args) throws Exception {
        Job job = Job.getInstance(getConf(), "candle");
        job.setJarByClass(this.getClass());

        Configuration conf = job.getConfiguration();

        job.setMapperClass(Map.class);
        job.setCombinerClass(Combine.class);
        job.setPartitionerClass(Partition.class);
        job.setReducerClass(Reduce.class);
        job.setNumReduceTasks(Integer.parseInt(conf.get("candle.num.reducers", CANDLE_NUM_REDUCER)));

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        job.setMapOutputKeyClass(Key.class);
        job.setMapOutputValueClass(Value.class);
        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(Text.class);

        return job.waitForCompletion(true) ? 0 : 1;
    }

    public static class Key implements WritableComparable<Key> {
        private int day;
        private int candleNum;

        public Key() {}

        public Key(int day, int candleNum) {
            this.day = day;
            this.candleNum = candleNum;
        }

        @Override
        public void write(DataOutput out) throws IOException {
            out.writeInt(day);
            out.writeInt(candleNum);
        }

        @Override
        public void readFields(DataInput in) throws IOException {
            day = in.readInt();
            candleNum = in.readInt();
        }

        @Override
        public int compareTo(Key o) {
            int res = Integer.compare(day, o.day);
            if (res != 0) {
                return res;
            } else {
                return Integer.compare(candleNum, o.candleNum);
            }
        }
    }

    public static class Value implements Writable {
        private int time;
        private double price;
        private long id;
        private String symbol;

        public Value() {}

        public Value(int time, double price, long id, String symbol) {
            this.time = time;
            this.price = price;
            this.id = id;
            this.symbol = symbol;
        }

        @Override
        public void write(DataOutput out) throws IOException {
            out.writeInt(time);
            out.writeDouble(price);
            out.writeLong(id);
            out.writeUTF(symbol);
        }

        @Override
        public void readFields(DataInput in) throws IOException {
            time = in.readInt();
            price = in.readDouble();
            id = in.readLong();
            symbol = in.readUTF();
        }
    }

    public static class Map extends Mapper<LongWritable, Text, Key, Value> {
        private int widthMillisec;
        private int timeFromMillisec;
        private int timeToMillisec;
        private String dateFromStr;
        private int dateFromInt;
        private Calendar dateFrom;
        private String dateToStr;
        private int dateToInt;
        String regex;

        @Override
        protected void setup(Context context) throws IOException {
            Configuration conf = context.getConfiguration();
            widthMillisec = Integer.parseInt(conf.get("candle.width", CANDLE_WIDTH));
            String timeFromStr = conf.get("candle.time.from", CANDLE_TIME_FROM);
            timeFromMillisec = Integer.parseInt(timeFromStr.substring(0, 2)) * 3600 * 1000
                    + Integer.parseInt(timeFromStr.substring(2)) * 60 * 1000;
            String timeToStr = conf.get("candle.time.to", CANDLE_TIME_TO);
            timeToMillisec = Integer.parseInt(timeToStr.substring(0, 2)) * 3600 * 1000
                    + Integer.parseInt(timeToStr.substring(2)) * 60 * 1000;
            dateFromStr = conf.get("candle.date.from", CANDLE_DATE_FROM);
            dateFromInt = Integer.parseInt(dateFromStr);
            dateFrom = new GregorianCalendar(
                dateFromInt / 10000,
                (dateFromInt / 100) % 100 - 1,
                dateFromInt % 100
            );
            dateFrom.setTimeZone(TimeZone.getTimeZone("GMT"));
            dateToStr = conf.get("candle.date.to", CANDLE_DATE_TO);
            dateToInt = Integer.parseInt(dateToStr);
            regex = conf.get("candle.securities", CANDLE_SECURITIES);
        }

        @Override
        public void map(LongWritable offset, Text lineText, Context context)
                throws IOException, InterruptedException {
            String line = lineText.toString();
            if (line.charAt(0) == '#') {
                return;
            }
            String[] commaSplit = line.split(",");

            String symbol = commaSplit[0];
            if (!Pattern.matches(regex, symbol)) {
                return;
            }

            String dateStr = commaSplit[2].substring(0, 8);
            int dateInt = Integer.parseInt(dateStr);
            if ((dateInt < dateFromInt) || (dateInt >= dateToInt)) {
                return;
            }
            Calendar date = new GregorianCalendar(
                dateInt / 10000,
                (dateInt / 100) % 100 - 1,
                dateInt % 100
            );
            date.setTimeZone(TimeZone.getTimeZone("GMT"));
            int day = (int)((date.getTime().getTime() - dateFrom.getTime().getTime()) / (24 * 3600 * 1000));

            String timeStr = commaSplit[2].substring(8);
            int timeMillisec = Integer.parseInt(timeStr.substring(0, 2)) * 3600 * 1000
                    + Integer.parseInt(timeStr.substring(2,4)) * 60 * 1000
                    + Integer.parseInt(timeStr.substring(4,6)) * 1000
                    + Integer.parseInt(timeStr.substring((6)));
            if ((timeMillisec < timeFromMillisec) || (timeMillisec >= timeToMillisec)) {
                return;
            }

            int candleNum = (timeMillisec - timeFromMillisec) / widthMillisec;

            long id = Long.parseLong(commaSplit[3]);
            double price = Double.parseDouble(commaSplit[4]);

            Key k = new Key(day, candleNum);
            Value v = new Value(timeMillisec, price, id, symbol);

            context.write(k, v);
        }
    }

    public static class CandleStruct {
        public int maxPriceTimeMillisec;
        public double maxPrice;
        public long maxPriceId;

        public int minPriceTimeMillisec;
        public double minPrice;
        public long minPriceId;

        public int minTimeMillisec;
        public double minTimePrice;
        public long minTimeId;

        public int maxTimeMillisec;
        public double maxTimePrice;
        public long maxTimeId;

        public CandleStruct(
                int maxPriceTimeMillisec, double maxPrice, long maxPriceId,
                int minPriceTimeMillisec, double minPrice, long minPriceId,
                int minTimeMillisec, double minTimePrice, long minTimeId,
                int maxTimeMillisec, double maxTimePrice, long maxTimeId
        ) {
            this.maxPriceTimeMillisec = maxPriceTimeMillisec;
            this.maxPrice = maxPrice;
            this.maxPriceId = maxPriceId;
            this.minPriceTimeMillisec = minPriceTimeMillisec;
            this.minPrice = minPrice;
            this.minPriceId = minPriceId;
            this.minTimeMillisec = minTimeMillisec;
            this.minTimePrice = minTimePrice;
            this.minTimeId = minTimeId;
            this.maxTimeMillisec = maxTimeMillisec;
            this.maxTimePrice = maxTimePrice;
            this.maxTimeId = maxTimeId;
        }
    }

    public static class Combine extends Reducer<Key, Value, Key, Value> {
        @Override
        public void reduce(Key k, Iterable<Value> values, Context context)
                throws IOException, InterruptedException {
            HashMap<String, CandleStruct> candles = new HashMap<>();
            for (Value v: values) {
                if (!candles.containsKey(v.symbol)) {
                    candles.put(
                        v.symbol,
                        new CandleStruct(
                            v.time, v.price, v.id,
                            v.time, v.price, v.id,
                            v.time, v.price, v.id,
                            v.time, v.price, v.id
                        )
                    );
                } else {
                    CandleStruct candle = candles.get(v.symbol);
                    if (candle.minPrice > v.price) {
                        candle.minPriceTimeMillisec = v.time;
                        candle.minPrice = v.price;
                        candle.minPriceId = v.id;
                    }
                    if (candle.maxPrice < v.price) {
                        candle.maxPriceTimeMillisec = v.time;
                        candle.maxPrice = v.price;
                        candle.maxPriceId = v.id;
                    }
                    if ((candle.minTimeMillisec > v.time) || ((candle.minTimeMillisec == v.time) && (candle.minTimeId > v.id))) {
                        candle.minTimeMillisec = v.time;
                        candle.minTimePrice = v.price;
                        candle.minTimeId = v.id;
                    }
                    if ((candle.maxTimeMillisec < v.time) || ((candle.maxTimeMillisec == v.time) && (candle.maxTimeId < v.id))) {
                        candle.maxTimeMillisec = v.time;
                        candle.maxTimePrice = v.price;
                        candle.maxTimeId = v.id;
                    }
                }
            }

            for (java.util.Map.Entry<String, CandleStruct> e: candles.entrySet()) {
                CandleStruct candle = e.getValue();
                String symbol = e.getKey();
                context.write(
                    k,
                    new Value(
                        candle.minTimeMillisec, candle.minTimePrice,
                        candle.minTimeId, symbol
                    )
                );
                context.write(
                    k,
                    new Value(
                        candle.maxTimeMillisec, candle.maxTimePrice,
                        candle.maxTimeId, symbol
                    )
                );
                context.write(
                    k,
                    new Value(
                        candle.minPriceTimeMillisec, candle.minPrice,
                        candle.minPriceId, symbol
                    )
                );
                context.write(
                    k,
                    new Value(
                        candle.maxPriceTimeMillisec, candle.maxPrice,
                        candle.maxPriceId, symbol
                    )
                );
            }
        }
    }

    public static class Reduce extends Reducer<Key, Value, NullWritable, Text> {
        private NullWritable nw;
        private Calendar calendar;
        private int yearFrom, monthFrom, dayFrom;
        private int candleWidth;
        private MultipleOutputs<NullWritable, Text> out;
        int timeFromMillisec;
        StringBuilder sb;

        @Override
        public void setup(Context context) throws IOException {
            nw = NullWritable.get();
            int dateInt = Integer.parseInt(context.getConfiguration().get("candle.date.from", CANDLE_DATE_FROM));
            yearFrom = dateInt / 10000;
            monthFrom = (dateInt / 100) % 100;
            dayFrom = dateInt % 100;
            candleWidth = Integer.parseInt(context.getConfiguration().get("candle.width", CANDLE_WIDTH));
            out = new MultipleOutputs<>(context);
            String timeFromStr = context.getConfiguration().get("candle.time.from", CANDLE_TIME_FROM);
            timeFromMillisec = Integer.parseInt(timeFromStr.substring(0, 2)) * 3600 * 1000
                    + Integer.parseInt(timeFromStr.substring(2)) * 60 * 1000;
            sb = new StringBuilder();
        }

        @Override
        public void reduce(Key k, Iterable<Value> values, Context context)
                throws IOException, InterruptedException {
            HashMap<String, CandleStruct> candles = new HashMap<>();
            for (Value v: values) {
                if (!candles.containsKey(v.symbol)) {
                    candles.put(
                        v.symbol,
                        new CandleStruct(
                            v.time, v.price, v.id,
                            v.time, v.price, v.id,
                            v.time, v.price, v.id,
                            v.time, v.price, v.id
                        )
                    );
                } else {
                    CandleStruct candle = candles.get(v.symbol);
                    if (candle.minPrice > v.price) {
                        candle.minPriceTimeMillisec = v.time;
                        candle.minPrice = v.price;
                        candle.minPriceId = v.id;
                    }
                    if (candle.maxPrice < v.price) {
                        candle.maxPriceTimeMillisec = v.time;
                        candle.maxPrice = v.price;
                        candle.maxPriceId = v.id;
                    }
                    if ((candle.minTimeMillisec > v.time) || ((candle.minTimeMillisec == v.time) && (candle.minTimeId > v.id))) {
                        candle.minTimeMillisec = v.time;
                        candle.minTimePrice = v.price;
                        candle.minTimeId = v.id;
                    }
                    if ((candle.maxTimeMillisec < v.time) || ((candle.maxTimeMillisec == v.time) && (candle.maxTimeId < v.id))) {
                        candle.maxTimeMillisec = v.time;
                        candle.maxTimePrice = v.price;
                        candle.maxTimeId = v.id;
                    }
                }
            }

            for (java.util.Map.Entry<String, CandleStruct> e: candles.entrySet()) {
                CandleStruct candle = e.getValue();
                String symbol = e.getKey();
                sb.replace(0, sb.length(), symbol);
                sb.append(',');
                calendar = new GregorianCalendar(yearFrom, monthFrom - 1, dayFrom);
                calendar.add(Calendar.DAY_OF_MONTH, k.day);
                calendar.add(Calendar.MILLISECOND, timeFromMillisec + k.candleNum * candleWidth);

                sb.append(calendar.get(Calendar.YEAR));
                sb.append(String.format("%02d", calendar.get(Calendar.MONTH) + 1));
                sb.append(String.format("%02d", calendar.get(Calendar.DAY_OF_MONTH)));
                sb.append(String.format("%02d", calendar.get(Calendar.HOUR_OF_DAY)));
                sb.append(String.format("%02d", calendar.get(Calendar.MINUTE)));
                sb.append(String.format("%02d", calendar.get(Calendar.SECOND)));
                sb.append(String.format("%03d", calendar.get(Calendar.MILLISECOND)));
                sb.append(',');
                double price = BigDecimal.valueOf(candle.minTimePrice).setScale(1, BigDecimal.ROUND_HALF_UP).doubleValue();
                sb.append(price);
//                sb.append(String.format("%.1f", candle.minTimePrice).replace(',', '.'));
                sb.append(',');
                price = BigDecimal.valueOf(candle.maxPrice).setScale(1, BigDecimal.ROUND_HALF_UP).doubleValue();
                sb.append(price);
//                sb.append(String.format("%.1f", candle.maxPrice).replace(',', '.'));
                sb.append(',');
                price = BigDecimal.valueOf(candle.minPrice).setScale(1, BigDecimal.ROUND_HALF_UP).doubleValue();
                sb.append(price);
//                sb.append(String.format("%.1f", candle.minPrice).replace(',', '.'));
                sb.append(',');
                price = BigDecimal.valueOf(candle.maxTimePrice).setScale(1, BigDecimal.ROUND_HALF_UP).doubleValue();
                sb.append(price);
//                sb.append(String.format("%.1f", candle.maxTimePrice).replace(',', '.'));
                out.write(nw, new Text(sb.toString()), symbol);
            }
        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            out.close();
        }
    }

    public static class Partition extends Partitioner<Key, Value> {
        @Override
        public int getPartition(Key k, Value v, int reducersNum){
            return Math.abs((k.day + "_" + k.candleNum).hashCode()) % reducersNum;
        }
    }
}